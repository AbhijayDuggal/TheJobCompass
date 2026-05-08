"""
JobCompass — Pipeline Orchestrator
Manages server lifecycle, model loading/unloading, and real-time processing.
This is the single interface the Streamlit app uses.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional
import signal
import json
import requests

# ── Server process handles ─────────────────────────────────────────────────────
_vllm_ocr_process   = None   # port 8000 — Qwen2.5-VL-3B (vision, NB01)
_vllm_text_process  = None   # port 8001 — Qwen2.5-7B (text, NB02)
_opensearch_process = None   # docker container

HF_HOME    = os.environ.get("HF_HOME", "D:/huggingface_cache")
WSL_MODE   = sys.platform != "win32"   # running inside WSL2 or Linux directly

# Retrieval engine singleton — kept alive for the session
_retrieval_engine: Optional[object] = None
_engine_lock = threading.Lock()


# ── OpenSearch ─────────────────────────────────────────────────────────────────

def start_opensearch(progress_cb: Optional[Callable] = None) -> bool:
    """Start OpenSearch Docker container if not already running."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", "opensearch"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and "true" in result.stdout.lower():
            if progress_cb: progress_cb("OpenSearch container already running.")
            return True
    except Exception:
        pass

    if progress_cb: progress_cb("Starting OpenSearch Docker container...")
    try:
        subprocess.run([
            "docker", "run", "-d",
            "--name", "opensearch",
            "-p", "9200:9200",
            "-e", "discovery.type=single-node",
            "-e", "DISABLE_SECURITY_PLUGIN=true",
            "-e", 'OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g',
            "opensearchproject/opensearch:2.13.0"
        ], capture_output=True, text=True, timeout=30)
    except Exception:
        # Try docker start if container exists
        subprocess.run(["docker", "start", "opensearch"], capture_output=True, timeout=15)

    if progress_cb: progress_cb("Waiting for OpenSearch to be ready (30s)...")
    for _ in range(30):
        time.sleep(1)
        from step4_indexing import is_opensearch_healthy
        if is_opensearch_healthy():
            if progress_cb: progress_cb("OpenSearch is ready.")
            return True
    if progress_cb: progress_cb("WARNING: OpenSearch may not be ready yet.")
    return False


def stop_opensearch(progress_cb: Optional[Callable] = None):
    try:
        subprocess.run(["docker", "stop", "opensearch"], capture_output=True, timeout=20)
        if progress_cb: progress_cb("OpenSearch stopped.")
    except Exception as e:
        if progress_cb: progress_cb(f"Could not stop OpenSearch: {e}")


def is_opensearch_running() -> bool:
    from step4_indexing import is_opensearch_healthy
    return is_opensearch_healthy()


# ── vLLM servers ───────────────────────────────────────────────────────────────

# def _run_command_background(cmd: str):

#     if WSL_MODE:
#         return subprocess.Popen(
#             cmd,
#             shell=True,
#             preexec_fn=os.setsid,
#         )

#     return subprocess.Popen(
#         cmd,
#         shell=True,
#         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
#     )
def _run_command_background(cmd: str, use_wsl=False):

    if use_wsl:

        full_cmd = ["wsl", "bash", "-lc", cmd]

        return subprocess.Popen(
            full_cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    return subprocess.Popen(
        cmd,
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

def _terminate_process(proc, port=None):
    try:
        if port:
            subprocess.run(
                [
                    "wsl",
                    "bash",
                    "-lc",
                    f"fuser -k {port}/tcp"
                ],
                capture_output=True,
            )
        if proc:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
    except Exception as e:
        print(f"Terminate error: {e}")

def _check_port(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0

def wait_until_server_stops(port: int, timeout: int = 120):

    start = time.time()

    while time.time() - start < timeout:

        if not _check_port(port):
            return True

        time.sleep(2)

    return False

def wait_until_no_vllm_processes(port: int, timeout=120):

    start = time.time()

    while time.time() - start < timeout:

        result = subprocess.run(
            [
                "wsl",
                "bash",
                "-lc",
                f"lsof -i :{port}"
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return True

        time.sleep(2)

    return False

def is_vllm_ocr_running() -> bool:
    return _check_port(8000)

def start_vllm_ocr(progress_cb=None):

    global _vllm_ocr_process

    if is_vllm_ocr_running():
        if progress_cb:
            progress_cb("OCR server already running.")
        return True

    cmd = get_vllm_ocr_command()

    if progress_cb:
        progress_cb("Starting OCR server...")

    _vllm_ocr_process = _run_command_background(
        cmd,
        use_wsl=True,
    )

    ok = wait_for_server(
        8000,
        timeout=900,
        progress_cb=progress_cb,
    )

    if ok:
        if progress_cb:
            progress_cb("OCR server ready.")
    else:
        if progress_cb:
            progress_cb("OCR server failed to start.")

    return ok


def stop_vllm_ocr(progress_cb=None):

    global _vllm_ocr_process

    if progress_cb:
        progress_cb("Stopping OCR server...")

    _terminate_process(
        _vllm_ocr_process,
        port=8000
    )

    stopped = wait_until_server_stops(8000)
    wait_until_no_vllm_processes(8000)
    time.sleep(15)

    _vllm_ocr_process = None

    if progress_cb:

        if stopped:
            progress_cb("OCR server fully stopped.")
        else:
            progress_cb("OCR shutdown timeout.")

def is_vllm_text_running() -> bool:
    return _check_port(8001)

def start_vllm_text(progress_cb=None):

    global _vllm_text_process

    if is_vllm_text_running():
        if progress_cb:
            progress_cb("Text server already running.")
        return True

    cmd = get_vllm_text_command()

    if progress_cb:
        progress_cb("Starting text server...")

    _vllm_text_process = _run_command_background(
        cmd,
        use_wsl=True,
    )

    ok = wait_for_server(
        8001,
        timeout=900,
        progress_cb=progress_cb,
    )

    if ok:
        if progress_cb:
            progress_cb("Text server ready.")
    else:
        if progress_cb:
            progress_cb("Text server failed to start.")

    return ok


def stop_vllm_text(progress_cb=None):

    global _vllm_text_process

    if progress_cb:
        progress_cb("Stopping text server...")

    _terminate_process(
        _vllm_text_process,
        port=8001
    )

    stopped = wait_until_server_stops(8001)
    wait_until_no_vllm_processes(8001)
    time.sleep(15)

    _vllm_text_process = None

    if progress_cb:

        if stopped:
            progress_cb("Text server fully stopped.")
        else:
            progress_cb("Text shutdown timeout.")

# def get_vllm_ocr_command() -> str:
#     """Returns the shell command to start the OCR vLLM server."""
#     return (
#         f"python -m vllm.entrypoints.openai.api_server "
#         f"--model Qwen/Qwen2.5-VL-3B-Instruct "
#         f"--trust-remote-code "
#         f"--dtype float16 "
#         f"--quantization bitsandbytes "
#         f"--load-format bitsandbytes "
#         f"--max-model-len 6144 "
#         f"--max-num-seqs 2 "
#         f"--gpu-memory-utilization 0.72 "
#         f"--limit-mm-per-prompt '{{\"image\": 1}}' "
#         f"--mm-processor-kwargs '{{\"min_pixels\": 200704, \"max_pixels\": 1254400}}' "
#         f"--port 8000 "
#         f"--host 0.0.0.0"
#     )

def get_vllm_ocr_command() -> str:

    return (
        "source /home/abhijay/vllm_env/bin/activate && "
        "python -m vllm.entrypoints.openai.api_server "
        "--model Qwen/Qwen2.5-VL-3B-Instruct "
        "--trust-remote-code "
        "--dtype float16 "
        "--quantization bitsandbytes "
        "--load-format bitsandbytes "
        "--max-model-len 6144 "
        "--max-num-seqs 2 "
        "--gpu-memory-utilization 0.72 "
        "--limit-mm-per-prompt '{\"image\":1}' "
        "--mm-processor-kwargs '{\"min_pixels\":200704,\"max_pixels\":1254400}' "
        "--port 8000 "
        "--host 0.0.0.0"
    )

# def get_vllm_text_command() -> str:
#     """Returns the shell command to start the text vLLM server."""
#     return (
#         f"python -m vllm.entrypoints.openai.api_server "
#         f"--model /mnt/d/models/Qwen2.5-7B-Instruct "
#         f"--trust-remote-code "
#         f"--dtype float16 "
#         f"--max-model-len 8192 "
#         f"--max-num-seqs 16 "
#         f"--gpu-memory-utilization 0.85 "
#         f"--port 8001 "
#         f"--host 0.0.0.0"
#     )
def get_vllm_text_command() -> str:

    return (
        f"source /home/abhijay/vllm_env/bin/activate && "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model /mnt/d/models/qwen-3b "
        f"--trust-remote-code "
        f"--dtype float16 "
        f"--quantization bitsandbytes "
        f"--load-format bitsandbytes "
        f"--max-model-len 4096 "
        f"--max-num-seqs 2 "
        f"--gpu-memory-utilization 0.68 "
        f"--port 8001 "
        f"--host 0.0.0.0"
    )

def wait_for_server(port: int, timeout: int = 120, progress_cb: Optional[Callable] = None) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if _check_port(port):
            return True
        time.sleep(2)
        if progress_cb:
            elapsed = int(time.time() - start)
            progress_cb(f"Waiting for server on port {port}... ({elapsed}s)")
    return False


# ── Retrieval engine management ────────────────────────────────────────────────

def get_retrieval_engine(progress_cb: Optional[Callable] = None):
    """Get or create the singleton retrieval engine."""
    global _retrieval_engine
    with _engine_lock:
        if _retrieval_engine is None:
            if progress_cb: progress_cb("Initialising retrieval engine...")
            sys.path.insert(0, str(Path(__file__).parent))
            from src.retrieval_engine import RetrievalEngine
            _retrieval_engine = RetrievalEngine(hf_home=HF_HOME)
            if progress_cb: progress_cb("Retrieval engine ready.")
        return _retrieval_engine


def unload_retrieval_models(progress_cb: Optional[Callable] = None):
    """Unload embed + reranker from VRAM. Safe to call anytime."""
    global _retrieval_engine
    with _engine_lock:
        if _retrieval_engine is not None:
            _retrieval_engine.unload_all()
            if progress_cb: progress_cb("Retrieval models unloaded from VRAM.")


# ── Full pipeline (batch) ──────────────────────────────────────────────────────

def run_full_pipeline(progress_cb: Optional[Callable] = None):
    """
    Run steps 1-4 sequentially.
    Steps 1+2 require vLLM servers (started externally).
    Step 3 loads/unloads embedding model.
    Step 4 requires OpenSearch.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Step 1
    if progress_cb: progress_cb("━━━ STEP 1: Resume Extraction ━━━")
    from step1_resume_extract import run as run_step1
    run_step1(progress_cb)

    # Step 2
    if progress_cb: progress_cb("━━━ STEP 2: Profile & Job Extraction ━━━")
    from step2_profile_extraction import run as run_step2
    run_step2(progress_cb)

    # Step 3
    if progress_cb: progress_cb("━━━ STEP 3: Embedding Generation ━━━")
    from step3_embedding import run as run_step3
    run_step3(progress_cb)

    # Step 4
    if progress_cb: progress_cb("━━━ STEP 4: OpenSearch Indexing ━━━")
    from step4_indexing import run as run_step4
    run_step4(progress_cb)

    if progress_cb: progress_cb("✅ Full pipeline complete!")


# ── Real-time single resume processing ────────────────────────────────────────

def process_new_resume(
    pdf_path: str,
    candidate_id: str,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Full pipeline for a single uploaded resume:
    1. Extract text (requires vLLM OCR server on 8000)
    2. Extract profile (requires vLLM text server on 8001)
    3. Embed (loads/unloads embedding model)
    4. Index in OpenSearch
    Returns structured candidate record.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Step 1: OCR
    if not is_vllm_ocr_running():
        start_vllm_ocr(progress_cb)
    if progress_cb: progress_cb("Extracting text from PDF...")
    from step1_resume_extract import run_single_pdf
    raw_record = run_single_pdf(pdf_path, candidate_id, progress_cb)
    stop_vllm_ocr(progress_cb)
    # progress_cb("Waiting for VRAM cleanup...")
    # time.sleep(15)

    # Step 2: Profile extraction
    if not is_vllm_text_running():
        start_vllm_text(progress_cb)
    if progress_cb: progress_cb("Extracting structured profile...")
    from step2_profile_extraction import extract_single_resume
    candidate = extract_single_resume(raw_record, progress_cb)
    stop_vllm_text(progress_cb)

    # Step 3: Embed (unloads retrieval engine models first)
    if progress_cb: progress_cb("Generating embedding...")
    unload_retrieval_models(progress_cb)
    from step3_embedding import embed_single_record, CANDIDATE_INSTRUCTION
    vec = embed_single_record(candidate["embedding_text"], CANDIDATE_INSTRUCTION, progress_cb)
    candidate["description_vector"] = vec

    # Step 4: Index
    if progress_cb: progress_cb("Indexing candidate...")
    from step4_indexing import index_single_candidate
    index_single_candidate(candidate, progress_cb)
    unload_retrieval_models(progress_cb)
    if progress_cb: progress_cb(f"Candidate {candidate_id} processed and indexed.")
    return {
        "candidate": candidate,
        "raw_record": raw_record,
    }


def process_new_job(
    job_data: dict,
    job_id: str,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Full pipeline for a new job description:
    1. Extract structured fields (requires vLLM text server on 8001)
    2. Embed
    3. Index
    Returns structured job record.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Step 2: Extract
    if not is_vllm_text_running():
        start_vllm_text(progress_cb)
    if progress_cb: progress_cb("Extracting structured job data...")
    from step2_profile_extraction import extract_single_job
    job = extract_single_job(job_data, job_id, progress_cb)
    stop_vllm_text(progress_cb)

    # Step 3: Embed
    if progress_cb: progress_cb("Generating job embedding...")
    unload_retrieval_models(progress_cb)
    from step3_embedding import embed_single_record, JOB_INSTRUCTION
    vec = embed_single_record(job["embedding_text"], JOB_INSTRUCTION, progress_cb)
    job["description_vector"] = vec

    # Step 4: Index
    if progress_cb: progress_cb("Indexing job...")
    from step4_indexing import index_single_job
    index_single_job(job, progress_cb)
    unload_retrieval_models(progress_cb)
    if progress_cb: progress_cb(f"Job {job_id} processed and indexed.")
    return {
        "job": job
    }


# ── Status helpers ─────────────────────────────────────────────────────────────

def get_system_status() -> dict:
    from step4_indexing import get_index_stats
    stats = get_index_stats()
    return {
        "opensearch"     : stats["healthy"],
        "vllm_ocr"       : is_vllm_ocr_running(),
        "vllm_text"      : is_vllm_text_running(),
        "candidates"     : stats["candidates"],
        "jobs"           : stats["jobs"],
        "data_ready"     : stats["candidates"] > 0 and stats["jobs"] > 0,
    }


def get_vram_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1024**3
            used  = torch.cuda.memory_allocated(0) / 1024**3
            free  = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            return {
                "available": True,
                "name": props.name,
                "total_gb": round(total, 1),
                "used_gb": round(used, 2),
                "free_gb": round(free, 2),
            }
    except Exception:
        pass
    return {"available": False}
