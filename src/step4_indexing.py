"""
JobCompass — Step 4: OpenSearch Indexing
Creates versioned indices with BM25 + HNSW kNN, bulk-loads all data.

Prerequisites: Docker container running
  docker run -d --name opensearch -p 9200:9200 \
    -e discovery.type=single-node \
    -e DISABLE_SECURITY_PLUGIN=true \
    -e OPENSEARCH_JAVA_OPTS="-Xms2g -Xmx2g" \
    opensearchproject/opensearch:2.13.0
"""

import os
import time
from pathlib import Path
from typing import Callable, Optional

import orjson
from opensearchpy import OpenSearch, RequestsHttpConnection
from tqdm.auto import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR    = "data/processed"
CANDIDATES_JSONL = f"{PROCESSED_DIR}/candidates_embedded.jsonl"
JOBS_JSONL       = f"{PROCESSED_DIR}/jobs_embedded.jsonl"

OS_HOST          = "localhost"
OS_PORT          = 9200

CANDIDATES_INDEX = "candidates_v1"
JOBS_INDEX       = "jobs_v1"
CANDIDATES_ALIAS = "candidates"
JOBS_ALIAS       = "jobs"

EMBEDDING_DIM    = 1024
HNSW_M               = 16
HNSW_EF_CONSTRUCTION = 100
BULK_BATCH_SIZE  = 500
FORCE_REINDEX    = False


# ── OpenSearch client ──────────────────────────────────────────────────────────

def get_client() -> OpenSearch:
    return OpenSearch(
        hosts            = [{"host": OS_HOST, "port": OS_PORT}],
        http_compress    = True,
        use_ssl          = False,
        verify_certs     = False,
        ssl_show_warn    = False,
        connection_class = RequestsHttpConnection,
        timeout          = 60,
        max_retries      = 3,
        retry_on_timeout = True,
    )


def is_opensearch_healthy() -> bool:
    try:
        c = get_client()
        health = c.cluster.health()
        return health["status"] in ("green", "yellow")
    except Exception:
        return False


# ── Index mappings (exact from NB04) ──────────────────────────────────────────

COMMON_SETTINGS = {
    "number_of_shards"  : 1,
    "number_of_replicas": 0,
    "knn"               : True,
    "knn.algo_param.ef_search": 100,
    "refresh_interval"  : "30s",
}

KNN_FIELD = {
    "type"      : "knn_vector",
    "dimension" : EMBEDDING_DIM,
    "method"    : {
        "name"      : "hnsw",
        "engine"    : "lucene",
        "space_type": "cosinesimil",
        "parameters": {
            "m"              : HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
        }
    }
}


def text_and_keyword(boost: float = 1.0) -> dict:
    return {
        "type"  : "text",
        "boost" : boost,
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
    }


CANDIDATES_BODY = {
    "settings": COMMON_SETTINGS,
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "candidate_id"          : {"type": "keyword"},
            "source_file"           : {"type": "keyword", "index": False},
            "source_category"       : {"type": "keyword", "index": False},
            "indexed_at"            : {"type": "date"},
            "extraction_method"     : {"type": "keyword", "index": False},
            "name"                  : {"type": "text"},
            "email"                 : {"type": "keyword"},
            "phone"                 : {"type": "keyword", "index": False},
            "location": {
                "type": "object",
                "properties": {
                    "city"   : {"type": "keyword"},
                    "country": {"type": "keyword"},
                }
            },
            "current_title"          : text_and_keyword(boost=2.0),
            "seniority"              : {"type": "keyword"},
            "total_years_experience" : {"type": "short"},
            "summary"                : {"type": "text"},
            "skills_flat": text_and_keyword(boost=3.0),
            "skills": {
                "type": "object",
                "properties": {
                    "technical" : text_and_keyword(boost=3.0),
                    "tools"     : text_and_keyword(boost=2.0),
                    "soft"      : {"type": "text"},
                    "languages" : {"type": "keyword"},
                }
            },
            "work_experience": {
                "type": "nested",
                "properties": {
                    "company"        : text_and_keyword(),
                    "title"          : {"type": "text"},
                    "start_date"     : {"type": "keyword"},
                    "end_date"       : {"type": "keyword"},
                    "duration_months": {"type": "short"},
                    "description"    : {"type": "text"},
                }
            },
            "education": {
                "type": "nested",
                "properties": {
                    "institution"    : text_and_keyword(),
                    "degree"         : {"type": "keyword"},
                    "field"          : {"type": "text"},
                    "graduation_year": {"type": "short"},
                }
            },
            "certifications"   : {"type": "keyword"},
            "languages_spoken" : {"type": "keyword"},
            "raw_text"         : {"type": "text"},
            "profile_quality_score" : {"type": "short"},
            "is_poor_quality"       : {"type": "boolean"},
            "missing_fields"        : {"type": "keyword"},
            "page_count"            : {"type": "short"},
            "embedding_text"   : {"type": "keyword", "index": False},
            "description_vector": KNN_FIELD,
        }
    }
}


JOBS_BODY = {
    "settings": COMMON_SETTINGS,
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "job_id"     : {"type": "keyword"},
            "indexed_at" : {"type": "date"},
            "posted_at"  : {"type": "date", "ignore_malformed": True},
            "is_active"  : {"type": "boolean"},
            "title"            : text_and_keyword(boost=2.0),
            "normalized_title" : text_and_keyword(boost=2.0),
            "role"             : {"type": "text"},
            "company"          : text_and_keyword(),
            "industry"         : {"type": "keyword"},
            "work_type"        : {"type": "keyword"},
            "company_size"     : {"type": "integer"},
            "seniority"        : {"type": "keyword"},
            "location": {
                "type": "object",
                "properties": {
                    "city"     : {"type": "keyword"},
                    "country"  : {"type": "keyword"},
                    "latitude" : {"type": "float"},
                    "longitude": {"type": "float"},
                }
            },
            "salary": {
                "type": "object",
                "properties": {
                    "min"     : {"type": "integer"},
                    "max"     : {"type": "integer"},
                    "currency": {"type": "keyword"},
                    "raw"     : {"type": "keyword", "index": False},
                }
            },
            "experience_required": {
                "type": "object",
                "properties": {
                    "min_years": {"type": "short"},
                    "max_years": {"type": "short"},
                    "raw"      : {"type": "keyword", "index": False},
                }
            },
            "qualifications" : {"type": "text"},
            "preference"     : {"type": "keyword"},
            "skills_flat"        : text_and_keyword(boost=3.0),
            "required_skills"    : text_and_keyword(boost=3.0),
            "nice_to_have_skills": {"type": "text"},
            "tech_stack"         : {"type": "keyword"},
            "domain_tags"        : {"type": "keyword"},
            "responsibilities_summary": {"type": "text"},
            "description"             : {"type": "text"},
            "responsibilities"        : {"type": "text"},
            "benefits"                : {"type": "text"},
            "raw_text"                : {"type": "text"},
            "company_profile" : {"type": "object", "enabled": False},
            "embedding_text"  : {"type": "keyword", "index": False},
            "job_portal"      : {"type": "keyword", "index": False},
            "contact_person"  : {"type": "keyword", "index": False},
            "description_vector": KNN_FIELD,
        }
    }
}


# ── Index lifecycle (exact from NB04) ─────────────────────────────────────────

def create_index_with_alias(client, index_name, alias_name, body, force=False) -> bool:
    exists = client.indices.exists(index=index_name)
    if exists and not force:
        count = client.count(index=index_name)["count"]
        if count > 0:
            return False
        client.indices.delete(index=index_name)
    elif exists and force:
        client.indices.delete(index=index_name)

    client.indices.create(index=index_name, body=body)
    actions = []
    if client.indices.exists_alias(name=alias_name):
        old_indices = list(client.indices.get_alias(name=alias_name).keys())
        for old in old_indices:
            if old != index_name:
                actions.append({"remove": {"index": old, "alias": alias_name}})
    actions.append({"add": {"index": index_name, "alias": alias_name}})
    client.indices.update_aliases(body={"actions": actions})
    return True


# ── Bulk indexing (exact from NB04) ───────────────────────────────────────────

def bulk_index(client, jsonl_path, index_name, id_field, needs_load, progress_cb=None) -> dict:
    if not needs_load:
        count = client.count(index=index_name)["count"]
        if progress_cb:
            progress_cb(f"  Skipped — {index_name} already has {count:,} documents.")
        return {"total": 0, "indexed": 0, "failed": 0, "elapsed_s": 0}

    records = []
    with open(jsonl_path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    if progress_cb:
        progress_cb(f"  Indexing {len(records):,} records into '{index_name}'")

    indexed, failed = 0, 0
    failed_samples  = []
    t_start         = time.time()
    pbar = tqdm(total=len(records), desc=f"  {index_name}", unit="doc")

    for batch_start in range(0, len(records), BULK_BATCH_SIZE):
        batch = records[batch_start : batch_start + BULK_BATCH_SIZE]
        lines = []
        for doc in batch:
            lines.append(orjson.dumps({"index": {"_index": index_name, "_id": str(doc[id_field])}}))
            clean = {k: v for k, v in doc.items() if not (k == "description_vector" and v is None)}
            lines.append(orjson.dumps(clean))
        body = b"\n".join(lines) + b"\n"
        try:
            resp = client.bulk(body=body)
            if resp.get("errors"):
                for item in resp["items"]:
                    op = item.get("index", {})
                    if op.get("error"):
                        failed += 1
                        if len(failed_samples) < 5:
                            failed_samples.append({"id": op.get("_id"), "reason": op["error"].get("reason","")[:120]})
                    else:
                        indexed += 1
            else:
                indexed += len(batch)
        except Exception as e:
            failed += len(batch)
            print(f"\n  Bulk request error: {e}")
        pbar.update(len(batch))

    pbar.close()
    client.indices.refresh(index=index_name)
    client.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "1s"}})

    elapsed = time.time() - t_start
    if progress_cb:
        progress_cb(f"  Indexed={indexed:,} | Failed={failed} | Time={elapsed:.1f}s")
    if failed_samples:
        for s in failed_samples:
            print(f"    [{s['id']}] {s['reason']}")

    return {"total": len(records), "indexed": indexed, "failed": failed, "elapsed_s": round(elapsed, 1)}


# ── Public API ─────────────────────────────────────────────────────────────────

def run(progress_cb: Optional[Callable] = None, force_reindex: bool = False):
    for p in [CANDIDATES_JSONL, JOBS_JSONL]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p} — run Step 3 first.")

    if not is_opensearch_healthy():
        raise RuntimeError("OpenSearch is not running. Start the Docker container first.")

    client = get_client()
    info   = client.info()
    if progress_cb:
        progress_cb(f"OpenSearch connected: v{info['version']['number']}")

    cand_needs_load = create_index_with_alias(client, CANDIDATES_INDEX, CANDIDATES_ALIAS, CANDIDATES_BODY, force_reindex)
    jobs_needs_load = create_index_with_alias(client, JOBS_INDEX, JOBS_ALIAS, JOBS_BODY, force_reindex)

    if progress_cb:
        progress_cb(f"Candidates index: {'created' if cand_needs_load else 'exists'}")
        progress_cb(f"Jobs index: {'created' if jobs_needs_load else 'exists'}")

    if progress_cb: progress_cb("=== Indexing candidates ===")
    bulk_index(client, CANDIDATES_JSONL, CANDIDATES_INDEX, "candidate_id", cand_needs_load, progress_cb)

    if progress_cb: progress_cb("=== Indexing jobs ===")
    bulk_index(client, JOBS_JSONL, JOBS_INDEX, "job_id", jobs_needs_load, progress_cb)

    cand_count = client.count(index=CANDIDATES_INDEX)["count"]
    jobs_count = client.count(index=JOBS_INDEX)["count"]

    if progress_cb:
        progress_cb(f"Step 4 complete. Candidates={cand_count:,} Jobs={jobs_count:,}")

    return {"candidates": cand_count, "jobs": jobs_count}


def index_single_candidate(candidate: dict, progress_cb: Optional[Callable] = None):
    """Index a single candidate record (real-time upload)."""
    client = get_client()
    cid = candidate["candidate_id"]
    clean = {k: v for k, v in candidate.items() if not (k == "description_vector" and v is None)}
    client.index(index=CANDIDATES_ALIAS, id=cid, body=clean)
    client.indices.refresh(index=CANDIDATES_ALIAS)
    if progress_cb:
        progress_cb(f"Candidate {cid} indexed.")


def index_single_job(job: dict, progress_cb: Optional[Callable] = None):
    """Index a single job record (real-time upload)."""
    client = get_client()
    jid = job["job_id"]
    clean = {k: v for k, v in job.items() if not (k == "description_vector" and v is None)}
    client.index(index=JOBS_ALIAS, id=jid, body=clean)
    client.indices.refresh(index=JOBS_ALIAS)
    if progress_cb:
        progress_cb(f"Job {jid} indexed.")


def get_candidate_by_id(candidate_id: str) -> Optional[dict]:
    try:
        client = get_client()
        return client.get(index=CANDIDATES_ALIAS, id=candidate_id)["_source"]
    except Exception:
        return None


def get_job_by_id(job_id: str) -> Optional[dict]:
    try:
        client = get_client()
        return client.get(index=JOBS_ALIAS, id=job_id)["_source"]
    except Exception:
        return None


def get_index_stats() -> dict:
    try:
        client = get_client()
        return {
            "candidates": client.count(index=CANDIDATES_ALIAS)["count"],
            "jobs": client.count(index=JOBS_ALIAS)["count"],
            "healthy": True,
        }
    except Exception:
        return {"candidates": 0, "jobs": 0, "healthy": False}
