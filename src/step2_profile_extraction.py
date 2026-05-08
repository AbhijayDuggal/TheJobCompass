"""
JobCompass — Step 2: Structured Profile & Job Extraction
Reads enriched_resumes.jsonl + job_descriptions.csv
Uses Qwen2.5-7B-Instruct via vLLM (port 8001) to extract structured data.
Outputs candidates.jsonl and jobs.jsonl
"""

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import aiohttp
import orjson
import pandas as pd
import requests
from tqdm.auto import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
ENRICHED_JSONL      = "data/extracted_text/enriched_resumes.jsonl"
JOBS_CSV            = "data/raw/job_descriptions.csv"
PROCESSED_DIR       = "data/processed"
CANDIDATES_JSONL    = f"{PROCESSED_DIR}/candidates.jsonl"
SKIPPED_CANDS_JSONL = f"{PROCESSED_DIR}/skipped_candidates.jsonl"
JOBS_JSONL          = f"{PROCESSED_DIR}/jobs.jsonl"
SKIPPED_JOBS_JSONL  = f"{PROCESSED_DIR}/skipped_jobs.jsonl"
TAXONOMY_PATH       = f"{PROCESSED_DIR}/skills_taxonomy.json"
PROGRESS_FILE       = f"{PROCESSED_DIR}/nb02_progress.json"

VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_MODEL    = "/mnt/d/models/qwen-3b"

CONCURRENT_REQUESTS  = 3
CHECKPOINT_EVERY     = 100
REQUEST_TIMEOUT_S    = 60
MAX_RETRIES          = 2
RETRY_BACKOFF_S      = 1.0

CANDIDATE_MAX_TOKENS = 900
JOB_MAX_TOKENS       = 700
RESUME_TEXT_TRUNCATE = 3500
JOB_TEXT_TRUNCATE    = 2500

MIN_RESUME_CHARS   = 150
MIN_JOB_DESC_CHARS = 80
MIN_SKILLS_COUNT   = 1

MAX_JOBS = 5000


# ── Utilities (exact from NB02) ────────────────────────────────────────────────

def _load_taxonomy():
    if Path(TAXONOMY_PATH).exists():
        with open(TAXONOMY_PATH, encoding="utf-8") as f:
            taxonomy_data = json.load(f)
        return {alias.lower().strip(): canonical
                for canonical, aliases in taxonomy_data.items()
                for alias in aliases}
    return {}

_alias_map = None

def get_alias_map():
    global _alias_map
    if _alias_map is None:
        _alias_map = _load_taxonomy()
    return _alias_map


def normalize_skill(raw: str) -> str:
    if not raw: return ""
    alias_map = get_alias_map()
    c = re.sub(r"\s+", " ", raw.strip().strip(".,;:()/[]{}")).strip()
    return "" if len(c) < 2 else alias_map.get(c.lower(), c)


def normalize_skills(lst: list) -> list:
    seen, out = set(), []
    for s in (lst or []):
        n = normalize_skill(str(s))
        if n and n.lower() not in seen:
            seen.add(n.lower()); out.append(n)
    return out


def parse_llm_json(raw: str) -> Optional[dict]:
    if not raw or not raw.strip(): return None
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    for attempt in [text, (re.search(r"\{.*\}", text, re.DOTALL) or type('', (), {'group': lambda s, x: None})()).group(0)]:
        if not attempt: continue
        try: return json.loads(attempt)
        except Exception: pass
    try:
        t = text
        t += "]" * max(0, t.count("[") - t.count("]"))
        t += "}" * max(0, t.count("{") - t.count("}"))
        return json.loads(t)
    except Exception:
        return None


def safe_str(val, default: str = "") -> str:
    if val is None: return default
    s = str(val).strip()
    return default if s.lower() in ("nan", "none", "") else s


def save_progress(progress):
    with open(PROGRESS_FILE, "wb") as f:
        f.write(orjson.dumps(progress))


def is_server_healthy() -> bool:
    try:
        r = requests.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Async vLLM client (exact from NB02) ───────────────────────────────────────

async def call_vllm(session: aiohttp.ClientSession, system_msg: str, user_msg: str, max_tokens: int) -> str:
    payload = {
        "model": VLLM_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
    }
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{VLLM_BASE_URL}/chat/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()

                    print("\n" + "=" * 80)
                    print("RAW LLM RESPONSE:")
                    print(content)
                    print("=" * 80 + "\n")
                    return content
                last_err = RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        except asyncio.TimeoutError:
            last_err = RuntimeError(f"Timeout after {REQUEST_TIMEOUT_S}s")
        except Exception as e:
            last_err = e
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF_S * (2 ** attempt))
    raise RuntimeError(f"vLLM failed after {MAX_RETRIES+1} attempts: {last_err}")


# ── Candidate extraction (exact from NB02) ────────────────────────────────────

CANDIDATE_SYSTEM = """You are a precise resume data extraction engine.
The input is OCR-extracted markdown from a scanned resume.
Return ONLY valid JSON. No markdown fences. No explanation. No preamble.
Use null for missing fields. Use [] for missing lists.
Extract only what is explicitly present in the text."""


def build_candidate_prompt(text: str) -> str:
    return f"""Extract structured data from this resume.

RESUME TEXT:
{text[:RESUME_TEXT_TRUNCATE]}

Return ONLY VALID JSON with EXACTLY these fields:
{{
  "name": "full name or null",
  "email": "email or null",
  "phone": "phone or null",
  "location": {{"city": "or null", "country": "or null"}},
  "current_title": "most recent job title or null",
  "summary": "professional summary or null",
  "total_years_experience": <integer or null>,
  "seniority": "fresher|junior|mid|senior|lead|executive or null",
  "skills": {{
    "technical": ["technical skills"],
    "soft": ["soft skills"],
    "tools": ["tools and software"],
    "languages": ["programming languages"]
  }},
  "work_experience": [
    {{"company": "name", "title": "job title", "start_date": "YYYY-MM or null",
      "end_date": "YYYY-MM or Present", "duration_months": <integer or null>,
      "description": "one sentence summary"}}
  ],
  "education": [
    {{"institution": "name", "degree": "Bachelor's etc",
      "field": "field or null", "graduation_year": <integer or null>}}
  ],
  "certifications": ["list"],
  "languages_spoken": ["spoken languages"]
  }}

  IMPORTANT:
  - Output ONLY valid JSON
  - No markdown
  - No explanations
  - No comments
  - No extra text
  """


def infer_seniority(years, title):
    t = (title or "").lower()
    for kw in ["executive","vp ","vice president","director","chief","cto","ceo","coo","cfo"]:
        if kw in t: return "executive"
    for kw in ["principal","staff ","lead ","head of","manager","architect"]:
        if kw in t: return "lead"
    for kw in ["senior","sr.","sr ","sr-"]:
        if kw in t: return "senior"
    for kw in ["junior","jr.","associate","entry level","entry-level"]:
        if kw in t: return "junior"
    for kw in ["intern","trainee","fresher","graduate trainee"]:
        if kw in t: return "fresher"
    if years is None: return "unknown"
    if years == 0: return "fresher"
    if years <= 2: return "junior"
    if years <= 5: return "mid"
    if years <= 9: return "senior"
    return "lead"


def compute_quality_score(profile, text):
    s = 0
    if profile.get("name"): s += 5
    if profile.get("email") or profile.get("phone"): s += 5
    if profile.get("current_title"): s += 10
    tech = (profile.get("skills") or {}).get("technical", [])
    s += 20 if len(tech) >= 3 else (10 if tech else 0)
    exp = profile.get("work_experience") or []
    s += 20 if len(exp) >= 2 else (12 if len(exp) == 1 else 0)
    if profile.get("education"): s += 10
    if profile.get("summary"): s += 10
    ar = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    s += int(min(ar * 1.5, 1.0) * 20)
    return min(s, 100)


def build_embedding_text_candidate(profile, content):
    parts = []
    if profile.get("current_title"): parts.append(profile["current_title"])
    sd = profile.get("skills") or {}
    all_skills = sd.get("technical",[]) + sd.get("tools",[]) + sd.get("languages",[])
    if all_skills: parts.append("Skills: " + ", ".join(all_skills[:30]))
    if profile.get("summary"): parts.append(profile["summary"][:300])
    for exp in (profile.get("work_experience") or [])[:3]:
        line = f"{exp.get('title','')} at {exp.get('company','')}: {exp.get('description','')}"
        parts.append(line[:200])
    return "\n".join(parts)[:1500]


def build_candidate_record(raw_record, profile):
    meta    = raw_record.get("metadata", {})
    content = raw_record.get("content", "")
    sd = profile.get("skills") or {}
    for key in ("technical","tools","soft","languages"):
        if sd.get(key): sd[key] = normalize_skills(sd[key])
    tech = sd.get("technical") or []
    tools = sd.get("tools") or []
    soft = sd.get("soft") or []
    skills_flat = list(dict.fromkeys(tech + tools + soft))
    years    = profile.get("total_years_experience")
    title    = profile.get("current_title")
    seniority = profile.get("seniority") or infer_seniority(years, title)

    missing_fields = []
    if not profile.get("name"): missing_fields.append("name")
    if not profile.get("email"): missing_fields.append("email")
    if not profile.get("phone"): missing_fields.append("phone")
    if not title: missing_fields.append("current_title")
    if not profile.get("summary"): missing_fields.append("summary")
    if not years: missing_fields.append("total_years_experience")
    if not skills_flat: missing_fields.append("skills")
    if not profile.get("education"): missing_fields.append("education")
    if not profile.get("work_experience"): missing_fields.append("work_experience")

    quality_score = compute_quality_score(profile, content)

    return {
        "candidate_id"          : raw_record["candidate_id"],
        "source_file"           : meta.get("pdf_path"),
        "source_category"       : meta.get("category"),
        "indexed_at"            : datetime.now(timezone.utc).isoformat(),
        "name"                  : profile.get("name"),
        "email"                 : profile.get("email"),
        "phone"                 : profile.get("phone"),
        "location"              : profile.get("location") or {},
        "current_title"         : title,
        "seniority"             : seniority,
        "total_years_experience": years,
        "summary"               : profile.get("summary"),
        "skills"                : sd,
        "skills_flat"           : skills_flat,
        "work_experience"       : profile.get("work_experience") or [],
        "education"             : profile.get("education") or [],
        "certifications"        : normalize_skills(profile.get("certifications") or []),
        "languages_spoken"      : profile.get("languages_spoken") or [],
        "raw_text"              : content,
        "page_count"            : meta.get("page_count", 1),
        "profile_quality_score" : quality_score,
        "is_poor_quality"       : quality_score < 50,
        "missing_fields"        : missing_fields,
        "extraction_method"     : "vllm-ocr+vllm-text-extraction",
        "embedding_text"        : build_embedding_text_candidate(profile, content),
        "description_vector"    : None,
    }


# ── Job extraction (exact from NB02) ──────────────────────────────────────────

JOB_SYSTEM = """You are a precise job listing data extraction engine.
Return ONLY valid JSON. No markdown fences. No explanation. No preamble.
Use null for missing fields. Use [] for missing lists.
Infer domain_tags purely from the job content — never use folder names or category labels."""


def build_job_prompt(title, role, desc, resp, skills_raw, experience):
    return f"""Extract structured data from this job listing.

JOB TITLE: {title}
ROLE: {role}
EXPERIENCE REQUIRED: {experience}
SKILLS LISTED: {skills_raw[:400]}
DESCRIPTION:
{desc[:JOB_TEXT_TRUNCATE]}
RESPONSIBILITIES:
{resp[:600]}

Return JSON with EXACTLY these fields:
{{
  "normalized_title"         : "concise standardized job title (e.g. 'Data Engineer', 'Frontend Developer')",
  "domain_tags"              : ["2-4 domain keywords inferred from content, e.g. 'data science', 'frontend', 'devops'"],
  "required_skills"          : ["skills that are clearly required"],
  "nice_to_have_skills"      : ["skills mentioned as preferred or bonus"],
  "tech_stack"               : ["specific technologies, frameworks, tools mentioned"],
  "seniority"                : "fresher|junior|mid|senior|lead|executive or null",
  "responsibilities_summary" : "2-3 sentence summary of what this person will do",
  "industry"                 : "industry sector or null"
}}"""


def clean_job_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"equal\s+opportunity\s+employer", "", text, flags=re.IGNORECASE)
    text = re.sub(r"all\s+qualified\s+applicants.{0,120}consideration", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{['\"]\w+['\"].*?\}", "", text, flags=re.DOTALL)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_skills_csv(raw) -> list:
    s = safe_str(raw)
    if not s: return []
    return [p.strip().strip("•·-").strip() for p in re.split(r"[\n,;/|]+", s) if len(p.strip()) > 1]


def parse_salary(raw) -> dict:
    result = {"min": None, "max": None, "currency": "USD", "raw": None}
    s = safe_str(raw)
    if not s: return result
    result["raw"] = s
    nums = [float(n) * (1000 if "k" in s.lower() else 1) for n in re.findall(r"[\d]+(?:\.\d+)?", s.replace(",","")) if n]
    if len(nums) >= 2: result["min"],result["max"] = int(nums[0]),int(nums[1])
    elif len(nums) == 1: result["min"] = result["max"] = int(nums[0])
    return result


def parse_experience(raw) -> dict:
    result = {"min_years": None, "max_years": None, "raw": None}
    s = safe_str(raw)
    if not s: return result
    result["raw"] = s
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2: result["min_years"],result["max_years"] = int(nums[0]),int(nums[1])
    elif len(nums) == 1: result["min_years"] = result["max_years"] = int(nums[0])
    return result


def parse_posted_date(raw) -> Optional[str]:
    s = safe_str(raw)
    if not s: return None
    for fmt in ("%d-%m-%Y","%Y-%m-%d","%m/%d/%Y","%d/%m/%Y","%Y/%m/%d"):
        try: return datetime.strptime(s, fmt).isoformat() + "Z"
        except: pass
    return None


def build_embedding_text_job(row, llm_data):
    parts = []
    t = llm_data.get("normalized_title") or safe_str(row.get("Job Title"))
    if t: parts.append(t)
    skills = (llm_data.get("required_skills") or []) + (llm_data.get("tech_stack") or [])
    if skills: parts.append("Skills: " + ", ".join(skills[:30]))
    if llm_data.get("responsibilities_summary"): parts.append(llm_data["responsibilities_summary"][:300])
    desc = clean_job_text(safe_str(row.get("Job Description","")))
    if desc: parts.append(desc[:400])
    return "\n".join(parts)[:1500]


def build_job_record(row, llm_data, job_id):
    title      = safe_str(row.get("Job Title"))
    role       = safe_str(row.get("Role"))
    clean_desc = clean_job_text(safe_str(row.get("Job Description","")))
    clean_resp = clean_job_text(safe_str(row.get("Responsibilities","")))
    skills_raw = safe_str(row.get("skills",""))
    csv_skills     = parse_skills_csv(skills_raw)
    req_skills     = normalize_skills(llm_data.get("required_skills") or csv_skills)
    nice_to_have   = normalize_skills(llm_data.get("nice_to_have_skills") or [])
    tech_stack     = normalize_skills(llm_data.get("tech_stack") or [])
    domain_tags = llm_data.get("domain_tags") or []
    if not domain_tags:
        norm = (llm_data.get("normalized_title") or title).lower()
        domain_tags = [w for w in re.split(r"\W+", norm) if len(w) > 3][:3]
    skills_flat = list(dict.fromkeys(req_skills + tech_stack))
    company_profile = {}
    raw_cp = safe_str(row.get("Company Profile",""))
    if raw_cp:
        try: company_profile = json.loads(raw_cp.replace("'", '"'))
        except: pass
    return {
        "job_id"                  : job_id,
        "indexed_at"              : datetime.now(timezone.utc).isoformat(),
        "is_active"               : True,
        "title"                   : title,
        "normalized_title"        : llm_data.get("normalized_title") or title,
        "role"                    : role,
        "company"                 : safe_str(row.get("Company")) or None,
        "company_profile"         : company_profile,
        "industry"                : llm_data.get("industry") or None,
        "work_type"               : safe_str(row.get("Work Type")) or None,
        "company_size"            : row.get("Company Size"),
        "location"                : {
            "city"     : safe_str(row.get("location")) or None,
            "country"  : safe_str(row.get("Country"))  or None,
            "latitude" : row.get("latitude"),
            "longitude": row.get("longitude"),
        },
        "salary"                  : parse_salary(row.get("Salary Range")),
        "experience_required"     : parse_experience(row.get("Experience")),
        "qualifications"          : safe_str(row.get("Qualifications")) or None,
        "preference"              : safe_str(row.get("Preference")) or None,
        "required_skills"         : req_skills,
        "nice_to_have_skills"     : nice_to_have,
        "tech_stack"              : tech_stack,
        "skills_flat"             : skills_flat,
        "domain_tags"             : domain_tags,
        "seniority"               : llm_data.get("seniority"),
        "responsibilities_summary": llm_data.get("responsibilities_summary"),
        "description"             : clean_desc,
        "responsibilities"        : clean_resp,
        "benefits"                : safe_str(row.get("Benefits")) or None,
        "raw_text"                : f"{title}\n{role}\n{clean_desc}\n{clean_resp}",
        "posted_at"               : parse_posted_date(row.get("Job Posting Date")),
        "job_portal"              : safe_str(row.get("Job Portal")) or None,
        "contact_person"          : safe_str(row.get("Contact Person")) or None,
        "embedding_text"          : build_embedding_text_job(row, llm_data),
        "description_vector"      : None,
    }


# ── Extraction runners ─────────────────────────────────────────────────────────

async def run_candidate_extraction(progress_cb: Optional[Callable] = None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "rb") as f:
            progress = orjson.loads(f.read())
    else:
        progress = {"candidates_done": [], "jobs_done": []}

    enriched_records = []
    with open(ENRICHED_JSONL, "rb") as f:
        for line in f:
            if line.strip(): enriched_records.append(orjson.loads(line))

    done_cand_ids   = set(progress.get("candidates_done", []))
    remaining_cands = [r for r in enriched_records if r["candidate_id"] not in done_cand_ids]

    if progress_cb:
        progress_cb(f"Candidate extraction: {len(remaining_cands)} remaining out of {len(enriched_records)}")

    start = time.time(); counts = {"written":0,"skipped":0,"error":0}; processed = 0
    lock = asyncio.Lock(); sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    out_fh  = open(CANDIDATES_JSONL,    "ab")
    skip_fh = open(SKIPPED_CANDS_JSONL, "ab")
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS+4)

    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(remaining_cands), desc="Candidates", unit="resume")

        async def _extract_one(raw_record):
            cid     = raw_record["candidate_id"]
            content = raw_record.get("content", "")
            
            #update
            content = content.replace("\\n", "\n")
            content = re.sub(r"\(cid:\d+\)", "", content)
            content = re.sub(r"\n+", "\n", content)
            content = content.strip()
            
            if len(content.strip()) < MIN_RESUME_CHARS:
                async with lock:
                    skip_fh.write(orjson.dumps({"candidate_id": cid, "reason": "text_too_short", "chars": len(content)}) + b"\n")
                return "skipped"
            raw_llm = await call_vllm(session, CANDIDATE_SYSTEM, build_candidate_prompt(content), CANDIDATE_MAX_TOKENS)
            profile = parse_llm_json(raw_llm)
            if not profile:
                profile = {
                    "name": None, "email": None, "phone": None, "location": {},
                    "current_title": None, "summary": None, "total_years_experience": None,
                    "seniority": None, "skills": {"technical":[],"soft":[],"tools":[],"languages":[]},
                    "work_experience": [], "education": [], "certifications": [], "languages_spoken": []
                }
            record = build_candidate_record(raw_record, profile)
            async with lock:
                out_fh.write(orjson.dumps(record) + b"\n")
                done_cand_ids.add(cid)
                progress["candidates_done"].append(cid)
            return "written"

        async def process_one(rec):
            nonlocal processed
            async with sem:
                try:
                    outcome = await _extract_one(rec)
                except Exception as e:
                    outcome = "error"
                    async with lock: print(f"\n  ERROR {rec['candidate_id']}: {e}")
                finally:
                    async with lock:
                        counts[outcome] = counts.get(outcome,0)+1; processed += 1
                        pbar.update(1)
                        if processed % CHECKPOINT_EVERY == 0:
                            out_fh.flush(); skip_fh.flush(); save_progress(progress)

        await asyncio.gather(*[process_one(r) for r in remaining_cands])
        pbar.close()

    out_fh.flush(); out_fh.close(); skip_fh.flush(); skip_fh.close()
    save_progress(progress)

    if progress_cb:
        progress_cb(f"Candidates: written={counts['written']}, skipped={counts['skipped']}, errors={counts['error']}")


async def run_job_extraction(progress_cb: Optional[Callable] = None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "rb") as f:
            progress = orjson.loads(f.read())
    else:
        progress = {"candidates_done": [], "jobs_done": []}

    if not Path(JOBS_CSV).exists():
        raise FileNotFoundError(f"Jobs CSV not found: {JOBS_CSV}")

    df = pd.read_csv(JOBS_CSV, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    for col in ["Job Title","Role","Job Description","skills","Responsibilities","Company"]:
        if col in df.columns: df[col] = df[col].fillna("").astype(str)
    mask = ~((df.get("Job Title", pd.Series(dtype=str)).str.strip() == "") &
             (df.get("Job Description", pd.Series(dtype=str)).str.strip() == ""))
    df = df[mask].copy()
    dedup_cols = [c for c in ["Job Title","Company","location"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    if MAX_JOBS is not None and len(df) > MAX_JOBS:
        df = df.iloc[:MAX_JOBS].reset_index(drop=True)

    job_manifest = []
    for idx, (_, row) in enumerate(df.iterrows()):
        job_id = f"job_{str(idx + 1).zfill(4)}"
        job_manifest.append((job_id, row))

    done_job_ids   = set(progress.get("jobs_done", []))
    remaining_jobs = [(jid, row) for jid, row in job_manifest if jid not in done_job_ids]

    if progress_cb:
        progress_cb(f"Job extraction: {len(remaining_jobs)} remaining out of {len(job_manifest)}")

    start = time.time(); counts = {"written":0,"skipped":0,"error":0}; processed = 0
    lock = asyncio.Lock(); sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    out_fh  = open(JOBS_JSONL,         "ab")
    skip_fh = open(SKIPPED_JOBS_JSONL, "ab")
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS+4)

    async def _extract_one_job(job_id, row, session):
        title      = safe_str(row.get("Job Title",""))
        role       = safe_str(row.get("Role",""))
        clean_desc = clean_job_text(safe_str(row.get("Job Description","")))
        clean_resp = clean_job_text(safe_str(row.get("Responsibilities","")))
        skills_raw = safe_str(row.get("skills",""))
        experience = safe_str(row.get("Experience",""))
        if len(clean_desc) < MIN_JOB_DESC_CHARS:
            async with lock:
                skip_fh.write(orjson.dumps({"job_id":job_id,"reason":"desc_too_short","chars":len(clean_desc)})+b"\n")
            return "skipped"
        raw_llm  = await call_vllm(session, JOB_SYSTEM,
                                   build_job_prompt(title,role,clean_desc,clean_resp,skills_raw,experience),
                                   JOB_MAX_TOKENS)
        llm_data = parse_llm_json(raw_llm) or {}
        record   = build_job_record(row, llm_data, job_id)
        if len(record["skills_flat"]) < MIN_SKILLS_COUNT:
            async with lock:
                skip_fh.write(orjson.dumps({"job_id":job_id,"reason":"no_skills"})+b"\n")
            return "skipped"
        async with lock:
            out_fh.write(orjson.dumps(record)+b"\n")
            done_job_ids.add(job_id)
            progress["jobs_done"].append(job_id)
        return "written"

    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(remaining_jobs), desc="Jobs", unit="job")

        async def process_one(job_id, row):
            nonlocal processed
            async with sem:
                try:
                    outcome = await _extract_one_job(job_id, row, session)
                except Exception as e:
                    outcome = "error"
                    async with lock: print(f"\n  ERROR {job_id}: {e}")
                finally:
                    async with lock:
                        counts[outcome] = counts.get(outcome,0)+1; processed += 1
                        pbar.update(1)
                        if processed % CHECKPOINT_EVERY == 0:
                            out_fh.flush(); skip_fh.flush(); save_progress(progress)

        await asyncio.gather(*[process_one(jid, row) for jid, row in remaining_jobs])
        pbar.close()

    out_fh.flush(); out_fh.close(); skip_fh.flush(); skip_fh.close()
    save_progress(progress)

    if progress_cb:
        progress_cb(f"Jobs: written={counts['written']}, skipped={counts['skipped']}, errors={counts['error']}")


def run(progress_cb: Optional[Callable] = None):
    if not is_server_healthy():
        raise RuntimeError("vLLM extraction server (port 8001) is not running. Start it first.")
    if not Path(ENRICHED_JSONL).exists():
        raise FileNotFoundError(f"enriched_resumes.jsonl not found — run Step 1 first.")

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    asyncio.run(run_candidate_extraction(progress_cb))
    asyncio.run(run_job_extraction(progress_cb))

    if progress_cb:
        progress_cb("Step 2 complete.")


def extract_single_resume(raw_record: dict, progress_cb: Optional[Callable] = None) -> dict:
    """Extract structured profile from a single enriched resume record (real-time use)."""
    if not is_server_healthy():
        raise RuntimeError("vLLM extraction server (port 8001) is not running.")

    async def _run():
        connector = aiohttp.TCPConnector(limit=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            content = raw_record.get("content", "")
            #update
            content = content.replace("\\n", "\n")
            content = re.sub(r"\(cid:\d+\)", "", content)
            content = re.sub(r"\n+", "\n", content)
            content = content.strip()
            
            raw_llm = await call_vllm(session, CANDIDATE_SYSTEM, build_candidate_prompt(content), CANDIDATE_MAX_TOKENS)
            profile = parse_llm_json(raw_llm)
            if not profile:
                profile = {
                    "name": None, "email": None, "phone": None, "location": {},
                    "current_title": None, "summary": None, "total_years_experience": None,
                    "seniority": None, "skills": {"technical":[],"soft":[],"tools":[],"languages":[]},
                    "work_experience": [], "education": [], "certifications": [], "languages_spoken": []
                }
            return build_candidate_record(raw_record, profile)

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    return asyncio.run(_run())


def extract_single_job(job_data: dict, job_id: str, progress_cb: Optional[Callable] = None) -> dict:
    """Extract structured job from a dict of job fields (real-time use)."""
    if not is_server_healthy():
        raise RuntimeError("vLLM extraction server (port 8001) is not running.")

    async def _run():
        connector = aiohttp.TCPConnector(limit=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            title      = job_data.get("Job Title", job_data.get("title", ""))
            role       = job_data.get("Role", job_data.get("role", ""))
            clean_desc = clean_job_text(job_data.get("Job Description", job_data.get("description", "")))
            clean_resp = clean_job_text(job_data.get("Responsibilities", job_data.get("responsibilities", "")))
            skills_raw = job_data.get("skills", "")
            experience = job_data.get("Experience", job_data.get("experience", ""))
            raw_llm  = await call_vllm(session, JOB_SYSTEM,
                                       build_job_prompt(title,role,clean_desc,clean_resp,skills_raw,experience),
                                       JOB_MAX_TOKENS)
            llm_data = parse_llm_json(raw_llm) or {}
            return build_job_record(job_data, llm_data, job_id)

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    return asyncio.run(_run())
