"""
JobCompass — Step 1: Resume Extraction
Reads PDFs from data/raw/resumes, uses vLLM OCR server (Qwen2.5-VL-3B),
outputs data/extracted_text/enriched_resumes.jsonl

NOTE: vLLM server must be started externally before calling run().
Server command: python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct --trust-remote-code \
    --dtype float16 --quantization bitsandbytes --load-format bitsandbytes \
    --max-model-len 6144 --max-num-seqs 2 --gpu-memory-utilization 0.72 \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 1254400}' \
    --port 8000 --host 0.0.0.0
"""

import asyncio
import base64
import gc
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import aiohttp
import fitz
import orjson
import pdfplumber
import requests
from PIL import Image
from tqdm.auto import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
RESUMES_DIR    = "data/raw/resumes"
OUTPUT_DIR     = "data/extracted_text"
ENRICHED_JSONL = f"{OUTPUT_DIR}/enriched_resumes.jsonl"
PROGRESS_FILE  = f"{OUTPUT_DIR}/ingestion_progress.json"

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL    = "Qwen/Qwen2.5-VL-3B-Instruct"

CONCURRENT_REQUESTS = 2
THREAD_WORKERS      = 1
CHECKPOINT_EVERY    = 10
CHUNK_SIZE          = 150

TEXT_MIN_CHARS      = 50
RENDER_DPI          = 170
MAX_RENDER_PIXELS   = 3_500_000
JPEG_QUALITY        = 82

REQUEST_TIMEOUT_S   = 120
RENDER_TIMEOUT_S    = 35
PDF_TASK_TIMEOUT_S  = 220

VLLM_MAX_RETRIES    = 1
RETRY_BACKOFF_S     = 1.5
HEALTH_TIMEOUT_S    = 5
COOLDOWN_EVERY_PDFS = 100
COOLDOWN_SECONDS    = 30

MM_MIN_PIXELS = 256 * 28 * 28
MM_MAX_PIXELS = 1600 * 28 * 28
MAX_NEW_TOKENS = 1600

VLLM_PROMPT = """You are doing OCR transcription of a resume page.

Return only text visibly present in the image.

Rules:
1. Do not invent names, companies, emails, skills, dates, or sections.
2. Do not summarize or paraphrase.
3. If the layout has two columns, read LEFT column top-to-bottom first, then RIGHT column top-to-bottom.
4. Preserve headings, bullets, and line breaks as much as possible.
5. Do not output markdown tables unless the source is an actual table.
6. Keep original language.
7. Output plain markdown text only. No explanations.
8. If unreadable, output exactly: [UNREADABLE_PAGE]
"""


# ── Extraction functions (exact from NB01) ─────────────────────────────────────

def clean_text(raw: str) -> str:
    if not raw:
        return ""
    lines = [
        ln.strip() for ln in raw.splitlines()
        if len(ln.strip()) > 1 and sum(c.isalnum() for c in ln) >= 2
    ]
    return "\n".join(lines)


def structure_pdfplumber_text(raw: str) -> str:
    if not raw:
        return ""
    out_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        is_heading = (
            stripped.isupper()
            and 2 <= len(stripped.split()) <= 6
            and len(stripped) < 60
        )
        if is_heading:
            out_lines.append(f"\n## {stripped.title()}\n")
        elif stripped.startswith(("•", "·", "▪", "◦", "-", "–", "—")):
            out_lines.append(f"- {stripped[1:].strip()}")
        else:
            out_lines.append(stripped)
    return "\n".join(out_lines).strip()


def extract_digital_pages(pdf_path: str) -> list[dict]:
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    raw = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                    text = structure_pdfplumber_text(raw)
                except Exception:
                    text = ""
                needs_vllm = len(text) < TEXT_MIN_CHARS
                results.append({
                    "page_num": i,
                    "text": text if not needs_vllm else "",
                    "char_count": len(text),
                    "needs_vllm": needs_vllm,
                    "method": "pdfplumber" if not needs_vllm else "pending",
                })
    except Exception as e:
        results = [{
            "page_num": 0, "text": "", "char_count": 0,
            "needs_vllm": True, "method": "pending", "error": str(e)
        }]
    return results


def render_page_to_base64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Optional[str]:
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num]
            rect = page.rect
            est_w = max(1, int(rect.width * dpi / 72.0))
            est_h = max(1, int(rect.height * dpi / 72.0))
            est_pixels = est_w * est_h
            if est_pixels > MAX_RENDER_PIXELS:
                scale = (MAX_RENDER_PIXELS / est_pixels) ** 0.5
                dpi = max(110, int(dpi * scale))
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img.close(); buf.close(); del pix
        fitz.TOOLS.store_shrink(100); gc.collect()
        return b64
    except Exception as e:
        print(f"  Render error {pdf_path} page {page_num}: {e}")
        fitz.TOOLS.store_shrink(100); gc.collect()
        return None


def parse_vllm_output(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return text.strip()


# ── Async LLM caller (exact from NB01) ────────────────────────────────────────

def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vllm_healthcheck(session: aiohttp.ClientSession) -> bool:
    base = VLLM_BASE_URL.replace("/v1", "")
    try:
        async with session.get(
            f"{base}/health",
            timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT_S),
        ) as r:
            return r.status == 200
    except Exception:
        return False


async def call_vllm_async(session: aiohttp.ClientSession, b64_image: str) -> str:
    payload = {
        "model": VLLM_MODEL,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                        "min_pixels": MM_MIN_PIXELS,
                        "max_pixels": MM_MAX_PIXELS,
                    },
                },
                {"type": "text", "text": VLLM_PROMPT},
            ],
        }],
    }
    last_err = None
    for attempt in range(VLLM_MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = data["choices"][0]["message"]["content"]
                    return parse_vllm_output(raw)
                body = await resp.text()
                last_err = RuntimeError(f"vLLM HTTP {resp.status}: {body[:300]}")
        except Exception as e:
            last_err = e
        if attempt < VLLM_MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"vLLM call failed: {last_err}")


def build_record(meta: dict, page_results: list[dict], elapsed_s: float) -> dict:
    num_pages = len(page_results)
    page_blocks = []
    for p in page_results:
        t = p["text"].strip()
        if t:
            page_blocks.append(
                f"<!-- page {p['page_num'] + 1} -->\n{t}" if num_pages > 1 else t
            )
    content = "\n\n---\n\n".join(page_blocks)
    methods_used = sorted(list({
        p["method"] for p in page_results if p["method"] not in ("pending", "")
    }))
    qwen_pages = sum(1 for p in page_results if p.get("method") == "vllm")
    return {
        "candidate_id": meta["candidate_id"],
        "metadata": {
            "pdf_name": meta["pdf_name"],
            "pdf_stem": meta["pdf_stem"],
            "pdf_path": meta["pdf_path"],
            "category": meta["category"],
            "page_count": num_pages,
            "digital_pages": num_pages - qwen_pages,
            "vllm_pages": qwen_pages,
            "total_chars": len(content),
            "methods_used": methods_used,
            "elapsed_s": round(elapsed_s, 2),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
        "content": content,
    }


async def process_pdf_async(meta, session, executor, loop) -> dict:
    t0 = time.time()
    page_results = await loop.run_in_executor(executor, extract_digital_pages, meta["pdf_path"])
    for p in page_results:
        if not p["needs_vllm"]:
            continue
        try:
            b64 = await asyncio.wait_for(
                loop.run_in_executor(executor, render_page_to_base64, meta["pdf_path"], p["page_num"], RENDER_DPI),
                timeout=RENDER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Render timeout: {meta['pdf_name']} page {p['page_num'] + 1}")
        if not b64:
            raise RuntimeError(f"Render failed: {meta['pdf_name']} page {p['page_num'] + 1}")
        txt = await call_vllm_async(session, b64)
        if not txt.strip():
            raise RuntimeError(f"Empty OCR: {meta['pdf_name']} page {p['page_num'] + 1}")
        p["text"] = txt; p["char_count"] = len(txt); p["method"] = "vllm"
        del b64; gc.collect()
    for p in page_results:
        if p["method"] == "pending":
            p["method"] = "skipped"; p["text"] = ""
    fitz.TOOLS.store_shrink(100); gc.collect()
    return build_record(meta, page_results, elapsed_s=time.time() - t0)


async def process_chunk(chunk, session, pbar, out_fh, processed_ids, progress_data, counters):
    queue = asyncio.Queue()
    for meta in chunk:
        queue.put_nowait(meta)
    lock = asyncio.Lock()
    pause_event = asyncio.Event()
    pause_event.set()
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        async def worker():
            while True:
                await pause_event.wait()
                try:
                    meta = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                do_cooldown = False
                cooldown_at = None
                try:
                    record = await asyncio.wait_for(
                        process_pdf_async(meta, session, executor, loop),
                        timeout=PDF_TASK_TIMEOUT_S,
                    )
                    async with lock:
                        out_fh.write(orjson.dumps(record) + b"\n")
                        cid = meta["candidate_id"]
                        if cid not in processed_ids:
                            processed_ids.add(cid)
                            progress_data["processed"].append(cid)
                        counters["written"] += 1
                except Exception as e:
                    msg = str(e)
                    async with lock:
                        counters["errors"] += 1
                        print(f"\nError {meta['pdf_name']}: {msg}")
                    if ("Cannot connect to host" in msg) or ("vLLM HTTP 500" in msg) or ("EngineCore" in msg):
                        raise RuntimeError("fatal_vllm")
                finally:
                    async with lock:
                        counters["processed"] += 1
                        elapsed = time.time() - counters["start"]
                        rate = max(counters["processed"], 1) / max(elapsed, 1)
                        remaining_n = counters["total"] - counters["processed"]
                        eta = remaining_n / max(rate, 0.001) / 60
                        pbar.update(1)
                        pbar.set_postfix({"written": counters["written"], "errors": counters["errors"], "ETA_min": f"{eta:.0f}"})
                        if counters["processed"] % CHECKPOINT_EVERY == 0:
                            out_fh.flush()
                            with open(PROGRESS_FILE, "wb") as pf:
                                pf.write(orjson.dumps(progress_data))
                        if counters["processed"] >= counters["next_cooldown_at"]:
                            do_cooldown = True
                            cooldown_at = counters["next_cooldown_at"]
                            counters["next_cooldown_at"] += COOLDOWN_EVERY_PDFS
                    queue.task_done()
                if do_cooldown:
                    pause_event.clear()
                    out_fh.flush()
                    with open(PROGRESS_FILE, "wb") as pf:
                        pf.write(orjson.dumps(progress_data))
                    print(f"\nCooldown: processed {cooldown_at} PDFs. Sleeping {COOLDOWN_SECONDS}s...")
                    await asyncio.sleep(COOLDOWN_SECONDS)
                    fitz.TOOLS.store_shrink(100); gc.collect()
                    pause_event.set()

        workers = [asyncio.create_task(worker()) for _ in range(CONCURRENT_REQUESTS)]
        await asyncio.gather(*workers)


async def run_ingestion(remaining, processed_ids, progress_data, progress_cb: Optional[Callable] = None):
    counters = {
        "written": 0, "errors": 0, "processed": 0,
        "start": time.time(), "total": len(remaining),
        "next_cooldown_at": COOLDOWN_EVERY_PDFS,
    }
    out_fh = open(ENRICHED_JSONL, "ab")
    pbar = tqdm(total=len(remaining), desc="Resumes", unit="pdf")
    try:
        for idx, chunk in enumerate(chunked(remaining, CHUNK_SIZE), start=1):
            connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS + 2)
            async with aiohttp.ClientSession(connector=connector) as session:
                if not await vllm_healthcheck(session):
                    raise RuntimeError("vLLM is not healthy. Restart server and rerun.")
                try:
                    await process_chunk(chunk, session, pbar, out_fh, processed_ids, progress_data, counters)
                except RuntimeError as e:
                    if str(e) == "fatal_vllm":
                        raise RuntimeError("Fatal vLLM failure detected. Stopping safely.")
            out_fh.flush()
            with open(PROGRESS_FILE, "wb") as pf:
                pf.write(orjson.dumps(progress_data))
            fitz.TOOLS.store_shrink(100); gc.collect()
            if progress_cb:
                progress_cb(f"Chunk {idx} complete. Written={counters['written']} Errors={counters['errors']}")
            print(f"\nChunk {idx} completed.")
    finally:
        out_fh.flush(); out_fh.close(); pbar.close()
        with open(PROGRESS_FILE, "wb") as pf:
            pf.write(orjson.dumps(progress_data))


def discover_pdfs() -> list[dict]:
    root = Path(RESUMES_DIR)
    pdf_list = sorted(root.rglob("*.pdf")) + sorted(root.rglob("*.PDF"))
    return [
        {
            "candidate_id": str(i + 1).zfill(4),
            "pdf_path": str(p),
            "pdf_name": p.name,
            "pdf_stem": p.stem,
            "category": p.parent.name,
        }
        for i, p in enumerate(pdf_list)
    ]


def is_server_healthy() -> bool:
    try:
        r = requests.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run(progress_cb: Optional[Callable] = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not is_server_healthy():
        raise RuntimeError("vLLM OCR server (port 8000) is not running. Start it first.")

    manifest = discover_pdfs()
    if not manifest:
        raise FileNotFoundError(f"No PDFs found under {RESUMES_DIR}")

    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "rb") as f:
            progress_data = orjson.loads(f.read())
        processed_ids = set(progress_data.get("processed", []))
    else:
        progress_data = {"processed": []}
        processed_ids = set()

    remaining = [m for m in manifest if m["candidate_id"] not in processed_ids]
    if progress_cb:
        progress_cb(f"Found {len(manifest)} PDFs. {len(processed_ids)} already done. {len(remaining)} remaining.")

    if not remaining:
        if progress_cb:
            progress_cb("All PDFs already processed. Nothing to do.")
        return

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    asyncio.run(run_ingestion(remaining, processed_ids, progress_data, progress_cb))

    if progress_cb:
        progress_cb(f"Step 1 complete. Output: {ENRICHED_JSONL}")


def run_single_pdf(pdf_path: str, candidate_id: str, progress_cb: Optional[Callable] = None) -> dict:
    """Process a single PDF file (for real-time candidate upload)."""
    if not is_server_healthy():
        raise RuntimeError("vLLM OCR server (port 8000) is not running.")

    meta = {
        "candidate_id": candidate_id,
        "pdf_path": pdf_path,
        "pdf_name": Path(pdf_path).name,
        "pdf_stem": Path(pdf_path).stem,
        "category": "uploaded",
    }

    async def _run():
        connector = aiohttp.TCPConnector(limit=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            return await process_pdf_async(meta, session, ThreadPoolExecutor(max_workers=1), asyncio.get_event_loop())

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    return asyncio.run(_run())
