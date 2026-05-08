"""
JobCompass — Step 3: Embedding Generation
Loads Qwen3-Embedding-0.6B, encodes all candidates and jobs.
Outputs candidates_embedded.jsonl and jobs_embedded.jsonl

IMPORTANT: Unloads model from VRAM after completion to free GPU for reranker.
"""

import gc
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR    = "data/processed"
CANDIDATES_IN    = f"{PROCESSED_DIR}/candidates.jsonl"
JOBS_IN          = f"{PROCESSED_DIR}/jobs.jsonl"
CANDIDATES_OUT   = f"{PROCESSED_DIR}/candidates_embedded.jsonl"
JOBS_OUT         = f"{PROCESSED_DIR}/jobs_embedded.jsonl"
PROGRESS_FILE    = f"{PROCESSED_DIR}/nb03_progress.json"

MODEL_NAME       = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM    = 1024
MAX_SEQ_LENGTH   = 512

CANDIDATE_INSTRUCTION = "Represent this resume for job matching"
JOB_INSTRUCTION       = "Represent this job description for candidate matching"

ENCODE_BATCH_SIZE = 128
CHECKPOINT_EVERY  = 128

HF_HOME = os.environ.get("HF_HOME", "D:/huggingface_cache")


# ── Model management ───────────────────────────────────────────────────────────

_tokenizer   = None
_embed_model = None
_device      = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(progress_cb: Optional[Callable] = None) -> int:
    """Load embedding model. Returns auto-tuned batch size."""
    global _tokenizer, _embed_model, _device
    os.environ["HF_HOME"] = HF_HOME
    _device = get_device()

    if progress_cb:
        progress_cb(f"Loading {MODEL_NAME} on {_device}...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)
    _embed_model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir   = HF_HOME,
        torch_dtype = torch.float16,
        device_map  = "auto",
    ).eval()

    torch.backends.cuda.enable_flash_sdp(False)

    # Auto-tune batch size based on free VRAM
    batch_size = ENCODE_BATCH_SIZE
    if _device == "cuda":
        torch.cuda.empty_cache()
        props   = torch.cuda.get_device_properties(0)
        free_gb = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
        if free_gb < 2.5:   batch_size = 8
        elif free_gb < 4.0: batch_size = 16
        elif free_gb < 5.5: batch_size = 32
        elif free_gb < 7.0: batch_size = 64
        else:                batch_size = 128
        if progress_cb:
            progress_cb(f"GPU: {props.name} | Free VRAM: {free_gb:.1f}GB | Batch size: {batch_size}")
    else:
        batch_size = 32
        if progress_cb:
            progress_cb("No GPU — running on CPU (slower)")

    return batch_size


def unload_model():
    global _tokenizer, _embed_model, _device
    if _embed_model is not None:
        del _embed_model
        _embed_model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def is_model_loaded() -> bool:
    return _embed_model is not None


# ── Encoding (exact from NB03) ─────────────────────────────────────────────────

def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size       = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths,
    ]


def encode_texts(texts: list[str], instruction: str, batch_size: int = ENCODE_BATCH_SIZE) -> np.ndarray:
    """Encode texts in micro-batches. Falls back to smaller sub-batches on OOM."""
    if _embed_model is None:
        raise RuntimeError("Embedding model not loaded. Call load_model() first.")

    formatted = [
        f"Instruct: {instruction}\nQuery: {t.strip()}" if t.strip() else instruction
        for t in texts
    ]

    all_embeddings = []
    micro_batch_size = batch_size

    i = 0
    while i < len(formatted):
        chunk = formatted[i : i + micro_batch_size]
        try:
            inputs = _tokenizer(
                chunk,
                max_length     = MAX_SEQ_LENGTH,
                padding        = True,
                truncation     = True,
                return_tensors = "pt",
            ).to(_embed_model.device)

            with torch.inference_mode():
                outputs = _embed_model(**inputs)

            embeddings = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())
            del inputs, outputs, embeddings
            if _device == "cuda":
                torch.cuda.empty_cache()
            i += micro_batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and micro_batch_size > 1:
                if _device == "cuda":
                    torch.cuda.empty_cache()
                micro_batch_size = max(1, micro_batch_size // 2)
                print(f"\n⚠ OOM — reducing micro_batch_size to {micro_batch_size} and retrying...")
            else:
                raise

    return np.vstack(all_embeddings)


def encode_single(text: str, instruction: str) -> np.ndarray:
    """Encode a single text. Returns L2-normalised float32 vector."""
    return encode_texts([text], instruction)[0]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "rb") as f:
            return orjson.loads(f.read())
    return {"candidates_done": [], "jobs_done": []}


def save_checkpoint(ckpt: dict):
    with open(PROGRESS_FILE, "wb") as f:
        f.write(orjson.dumps(ckpt))


# ── Core encode-and-write (exact from NB03) ────────────────────────────────────

def encode_and_write(
    records     : list[dict],
    id_field    : str,
    text_field  : str,
    instruction : str,
    output_path : str,
    done_ids    : set,
    ckpt_key    : str,
    checkpoint  : dict,
    batch_size  : int,
    progress_cb : Optional[Callable] = None,
) -> int:
    remaining = [r for r in records if r[id_field] not in done_ids]
    skipped   = len(records) - len(remaining)

    if progress_cb:
        progress_cb(f"  Total={len(records):,}  Already done={skipped:,}  To encode={len(remaining):,}")

    if not remaining:
        if progress_cb:
            progress_cb("  Nothing to do — all records already embedded.")
        return 0

    written = 0
    t_start = time.time()
    out_fh  = open(output_path, "ab")

    try:
        pbar = tqdm(total=len(remaining), desc="  Encoding", unit="rec")
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]
            texts = []
            for rec in batch:
                text = (rec.get(text_field) or "").strip()
                if not text:
                    text = (rec.get("raw_text") or "")[:800].strip()
                texts.append(text)

            vectors = encode_texts(texts, instruction, batch_size)
            batch_ids = []
            for rec, vec in zip(batch, vectors):
                out_rec = {**rec, "description_vector": vec.tolist()}
                out_fh.write(orjson.dumps(out_rec) + b"\n")
                batch_ids.append(rec[id_field])
                written += 1

            out_fh.flush()
            checkpoint[ckpt_key].extend(batch_ids)
            done_ids.update(batch_ids)
            save_checkpoint(checkpoint)
            pbar.update(len(batch))

            if _device == "cuda" and batch_start % (batch_size * 8) == 0:
                torch.cuda.empty_cache()

        pbar.close()

    finally:
        out_fh.close()

    elapsed = time.time() - t_start
    if progress_cb:
        progress_cb(f"  Written={written:,} | Time={elapsed:.1f}s | {written/max(elapsed,0.001):.0f} rec/s")
    return written


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(path: str) -> dict:
    records = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    issues = {"missing": 0, "wrong_dim": 0, "nan_inf": 0, "non_unit": 0}
    for rec in records:
        vec = rec.get("description_vector")
        if vec is None:
            issues["missing"] += 1; continue
        arr = np.array(vec, dtype=np.float32)
        if len(arr) != EMBEDDING_DIM:
            issues["wrong_dim"] += 1; continue
        if np.any(~np.isfinite(arr)):
            issues["nan_inf"] += 1; continue
        if not (0.98 < np.linalg.norm(arr) < 1.02):
            issues["non_unit"] += 1
    return {"total": len(records), **issues}


# ── Public API ─────────────────────────────────────────────────────────────────

def run(progress_cb: Optional[Callable] = None):
    """Run full embedding generation pipeline. Loads and unloads model."""
    for p in [CANDIDATES_IN, JOBS_IN]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p} — run Step 2 first.")

    batch_size = load_model(progress_cb)
    try:
        checkpoint    = load_checkpoint()
        candidates    = []
        with open(CANDIDATES_IN, "rb") as f:
            for line in f:
                if line.strip(): candidates.append(orjson.loads(line))
        jobs = []
        with open(JOBS_IN, "rb") as f:
            for line in f:
                if line.strip(): jobs.append(orjson.loads(line))

        if progress_cb:
            progress_cb(f"Loaded {len(candidates):,} candidates and {len(jobs):,} jobs")

        if progress_cb: progress_cb("=== Encoding candidates ===")
        done_cand_ids = set(checkpoint.get("candidates_done", []))
        encode_and_write(
            records=candidates, id_field="candidate_id", text_field="embedding_text",
            instruction=CANDIDATE_INSTRUCTION, output_path=CANDIDATES_OUT,
            done_ids=done_cand_ids, ckpt_key="candidates_done", checkpoint=checkpoint,
            batch_size=batch_size, progress_cb=progress_cb,
        )

        if progress_cb: progress_cb("=== Encoding jobs ===")
        done_job_ids = set(checkpoint.get("jobs_done", []))
        encode_and_write(
            records=jobs, id_field="job_id", text_field="embedding_text",
            instruction=JOB_INSTRUCTION, output_path=JOBS_OUT,
            done_ids=done_job_ids, ckpt_key="jobs_done", checkpoint=checkpoint,
            batch_size=batch_size, progress_cb=progress_cb,
        )

        if progress_cb: progress_cb("Step 3 complete. Unloading embedding model to free VRAM...")
    finally:
        unload_model()

    if progress_cb: progress_cb("Embedding model unloaded.")


def embed_single_record(text: str, instruction: str, progress_cb: Optional[Callable] = None) -> list[float]:
    """Embed a single text. Model must already be loaded or will be loaded temporarily."""
    was_loaded = is_model_loaded()
    if not was_loaded:
        load_model(progress_cb)
    try:
        vec = encode_single(text, instruction)
        return vec.tolist()
    finally:
        if not was_loaded:
            unload_model()
