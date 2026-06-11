# TheJobCompass
An AI-powered recruitment intelligence platform that automatically extracts, understands, indexes, and matches resumes and job descriptions using Large Language Models, hybrid search, and advanced reranking techniques.

## Overview

TheJobCompass is a bidirectional talent discovery and job recommendation system designed to improve recruitment search relevance beyond traditional keyword matching.

The platform processes resumes and job descriptions, converts them into structured profiles, generates semantic embeddings, indexes them into OpenSearch, and performs hybrid retrieval using both sparse and dense search techniques.

## System Architecture

```text
Resume PDF
    │
    ▼
Qwen-VL OCR Extraction
    │
    ▼
Structured Candidate Profile
    │
    ▼
Qwen Embedding Generation
    │
    ▼
OpenSearch Index

──────────────────────────────────────────

Job Description
    │
    ▼
LLM-Based Job Understanding
    │
    ▼
Structured Job Profile
    │
    ▼
Qwen Embedding Generation
    │
    ▼
OpenSearch Index

──────────────────────────────────────────

Search Request
    │
    ├── BM25 Retrieval
    ├── Semantic Vector Search
    │
    ▼
Reciprocal Rank Fusion (RRF)
    │
    ▼
BGE Cross-Encoder Reranking
    │
    ▼
Final Ranked Results
```

## Key Features

* Resume parsing and OCR using Qwen2.5-VL
* Structured candidate and job profile extraction
* Semantic embeddings using Qwen3-Embedding
* OpenSearch-based indexing and retrieval
* Hybrid Search (BM25 + Vector Search)
* Reciprocal Rank Fusion (RRF)
* BGE Cross-Encoder reranking
* Bidirectional matching:

  * Candidate → Jobs
  * Job → Candidates
* Streamlit-based interactive interface
* GPU-aware model loading and memory management

## Technology Stack

| Component                    | Technology                   |
| ---------------------------- | ---------------------------- |
| OCR & Document Understanding | Qwen2.5-VL-3B-Instruct       |
| Profile Extraction           | Qwen 3B                      |
| Embeddings                   | Qwen3-Embedding-0.6B         |
| Search Engine                | OpenSearch                   |
| Sparse Retrieval             | BM25                         |
| Dense Retrieval              | HNSW Vector Search           |
| Rank Fusion                  | Reciprocal Rank Fusion (RRF) |
| Reranking                    | BAAI BGE-Reranker-v2-m3      |
| Backend                      | Python                       |
| UI                           | Streamlit                    |
| Inference                    | vLLM                         |
| Infrastructure               | Docker, CUDA                 |

## Project Structure

```text
TheJobCompass/
│
├── app.py
│
├── src/
│   ├── pipeline_orchestrator.py
│   ├── step1_resume_extract.py
│   ├── step2_profile_extraction.py
│   ├── step3_embedding.py
│   ├── step4_indexing.py
│   ├── retrieval_engine.py
│   └── retrieval_engine_helpers.py
│
├── Notebooks/
│   ├── 1_resume_extract_final.ipynb
│   ├── 2_profile_and_job_extraction.ipynb
│   ├── 3_embedding_generation.ipynb
│   ├── 4_opensearch_indexing.ipynb
│   └── 5_retrieval_reranking.ipynb
│
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.10+
* CUDA-enabled GPU (8GB+ VRAM recommended)
* Docker
* OpenSearch
* vLLM

## Installation

### Clone Repository

```bash
git clone https://github.com/abhijayduggal/TheJobCompass.git

cd TheJobCompass
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / WSL:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## OpenSearch Setup

Run OpenSearch using Docker:

```bash
docker run -d \
  --name opensearch \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e DISABLE_SECURITY_PLUGIN=true \
  opensearchproject/opensearch:2.13.0
```

Verify:

```bash
curl http://localhost:9200
```

## Start vLLM Servers

The application requires two vLLM servers.

### OCR Server (Port 8000)

Used for resume OCR and document understanding.

```bash
source /home/user/vllm_env/bin/activate && \
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--trust-remote-code \
--dtype float16 \
--quantization bitsandbytes \
--load-format bitsandbytes \
--max-model-len 6144 \
--max-num-seqs 2 \
--gpu-memory-utilization 0.72 \
--limit-mm-per-prompt '{"image":1}' \
--mm-processor-kwargs '{"min_pixels":200704,"max_pixels":1254400}' \
--port 8000 \
--host 0.0.0.0
```

### Text Server (Port 8001)

Used for profile extraction and job understanding.

```bash
source /home/user/vllm_env/bin/activate && \
python -m vllm.entrypoints.openai.api_server \
--model /mnt/d/models/qwen-3b \
--trust-remote-code \
--dtype float16 \
--quantization bitsandbytes \
--load-format bitsandbytes \
--max-model-len 4096 \
--max-num-seqs 2 \
--gpu-memory-utilization 0.68 \
--port 8001 \
--host 0.0.0.0
```

## Launch Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Screenshots

### Candidate Dashboard

<img width="1877" height="897" alt="Screenshot 2026-06-11 192220" src="https://github.com/user-attachments/assets/5d0897d0-83a5-4c5f-8ae7-e59d670ee6bf" />

### Recruiter Dashboard

<img width="1870" height="901" alt="Screenshot 2026-06-11 192830" src="https://github.com/user-attachments/assets/0e3cd486-59cf-43a9-9aa0-172be6c6ec28" />



## Author
**Abhijay Duggal**



