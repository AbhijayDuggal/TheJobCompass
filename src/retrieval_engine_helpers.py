"""
JobCompass — Retrieval Engine Helpers
Helper functions for build_job_search_query, build_candidate_search_query,
rerank text builders, and result formatters.
These are imported by retrieval_engine.py (NB05 generated module).
"""

from typing import Optional


# ── Query builders ─────────────────────────────────────────────────────────────

def build_job_search_query(
    candidate: dict,
    filter_seniority: Optional[list] = None,
    filter_location: Optional[str]   = None,
    filter_work_type: Optional[str]  = None,
):
    """
    Build BM25 query + embedding text + filters for job search from candidate profile.
    Returns (embed_text, bm25_body, filters)
    """
    title       = candidate.get("current_title") or ""
    skills_flat = candidate.get("skills_flat") or []
    if not isinstance(skills_flat, list): skills_flat = []
    summary     = candidate.get("summary") or ""
    skills_str  = " ".join(skills_flat[:20])

    embed_text = "\n".join(filter(None, [
        title,
        f"Skills: {skills_str}" if skills_str else "",
        summary[:300],
    ]))[:1500]

    query_text = " ".join(filter(None, [title, skills_str, summary[:200]]))

    bm25_body = {
        "query": {
            "multi_match": {
                "query" : query_text,
                "fields": [
                    "title^3",
                    "normalized_title^3",
                    "required_skills^2",
                    "description",
                    "responsibilities_summary",
                ],
                "type": "best_fields",
            }
        }
    }

    filters = []
    if filter_seniority:
        filters.append({"terms": {"seniority": filter_seniority}})
    if filter_location:
        filters.append({"term": {"location.city": filter_location}})
    if filter_work_type:
        filters.append({"term": {"work_type": filter_work_type}})

    return embed_text, bm25_body, filters


def build_candidate_search_query(
    job: dict,
    filter_seniority: Optional[list] = None,
    filter_location: Optional[str]   = None,
):
    """
    Build BM25 query + embedding text + filters for candidate search from job.
    Returns (embed_text, bm25_body, filters)
    """
    title       = job.get("normalized_title") or job.get("title") or ""
    req_skills  = job.get("required_skills") or []
    if not isinstance(req_skills, list): req_skills = []
    skills_flat = job.get("skills_flat") or req_skills
    if not isinstance(skills_flat, list): skills_flat = []
    resp_summary = job.get("responsibilities_summary") or ""
    description  = job.get("description") or ""

    skills_str = " ".join(skills_flat[:20])

    embed_text = "\n".join(filter(None, [
        title,
        f"Skills: {skills_str}" if skills_str else "",
        resp_summary[:300],
        description[:200],
    ]))[:1500]

    query_text = " ".join(filter(None, [title, skills_str, resp_summary[:200]]))

    bm25_body = {
        "query": {
            "multi_match": {
                "query" : query_text,
                "fields": [
                    "current_title^3",
                    "skills_flat^2",
                    "raw_text",
                    "summary",
                ],
                "type": "best_fields",
            }
        }
    }

    filters = []
    if filter_seniority:
        filters.append({"terms": {"seniority": filter_seniority}})
    if filter_location:
        filters.append({"term": {"location.city": filter_location}})

    return embed_text, bm25_body, filters


# ── Rerank text builders ───────────────────────────────────────────────────────

def build_rerank_text_job(job_source: dict) -> str:
    """Build text representation of a job for cross-encoder reranking."""
    parts = []
    if job_source.get("normalized_title"):
        parts.append(job_source["normalized_title"])
    if job_source.get("required_skills"):
        parts.append("Required: " + ", ".join(job_source["required_skills"][:15]))
    if job_source.get("responsibilities_summary"):
        parts.append(job_source["responsibilities_summary"][:300])
    if job_source.get("description"):
        parts.append(job_source["description"][:300])
    return "\n".join(parts)[:1000]


def build_rerank_text_candidate(candidate_source: dict) -> str:
    """Build text representation of a candidate for cross-encoder reranking."""
    parts = []
    if candidate_source.get("current_title"):
        parts.append(candidate_source["current_title"])
    skills = candidate_source.get("skills_flat") or []
    if skills:
        parts.append("Skills: " + ", ".join(skills[:15]))
    if candidate_source.get("summary"):
        parts.append(candidate_source["summary"][:300])
    for exp in (candidate_source.get("work_experience") or [])[:2]:
        desc = exp.get("description", "")
        if desc:
            parts.append(f"{exp.get('title','')} at {exp.get('company','')}: {desc[:150]}")
    return "\n".join(parts)[:1000]


# ── Result formatters ──────────────────────────────────────────────────────────

from rapidfuzz import fuzz


def _compute_matched_skills(candidate_skills, job_skills, threshold=85):

    cand_norm = [
        s.strip().lower()
        for s in (candidate_skills or [])
        if isinstance(s, str)
    ]

    matched = []
    missing = []

    for js in (job_skills or []):

        if not isinstance(js, str):
            continue

        js_norm = js.strip().lower()

        best_score = max(
            (fuzz.ratio(js_norm, cs) for cs in cand_norm),
            default=0
        )

        if best_score >= threshold:
            matched.append(js)
        else:
            missing.append(js)

    return matched, missing


def format_job_results(reranked: list[dict], candidate: dict) -> list[dict]:
    """Format reranked jobs into the response structure for a candidate."""
    candidate_skills = candidate.get("skills_flat") or []
    results = []
    for rank, item in enumerate(reranked, 1):
        src = item["source"]
        req_skills = (
            src.get("required_skills")
            or src.get("skills_flat")
            or src.get("domain_tags")
            or []
        )
        matched, missing = _compute_matched_skills(candidate_skills, req_skills)

        results.append({
            "rank"            : rank,
            "job_id"          : item["id"],
            "rerank_score"    : round(item.get("rerank_score", 0.0), 4),
            "rrf_score"       : round(item.get("rrf_score", 0.0), 6),
            "bm25_rank"       : item.get("bm25_rank"),
            "semantic_rank"   : item.get("semantic_rank"),

            "title"           : src.get("normalized_title") or src.get("title"),
            "company"         : src.get("company"),
            "location"        : src.get("location", {}),
            "seniority"       : src.get("seniority"),
            "work_type"       : src.get("work_type"),
            "industry"        : src.get("industry"),
            "salary"          : src.get("salary", {}),
            "experience_required": src.get("experience_required", {}),

            "required_skills" : req_skills,
            "matched_skills"  : matched,
            "missing_skills"  : missing,
            "match_pct"       : round(len(matched) / max(len(req_skills), 1) * 100, 1),

            "responsibilities_summary": src.get("responsibilities_summary"),
            "description"     : (src.get("description") or "")[:500],
            "benefits"        : src.get("benefits"),
            "posted_at"       : src.get("posted_at"),
            "domain_tags"     : src.get("domain_tags") or [],
        })
    return results


def format_candidate_results(reranked: list[dict], job: dict) -> list[dict]:
    """Format reranked candidates into the response structure for a recruiter."""
    job_skills = (job.get("required_skills") or []) + (job.get("tech_stack") or [])
    results = []
    for rank, item in enumerate(reranked, 1):
        src = item["source"]
        cand_skills = src.get("skills_flat") or []
        matched, missing = _compute_matched_skills(cand_skills, job_skills)

        results.append({
            "rank"            : rank,
            "candidate_id"    : item["id"],
            "rerank_score"    : round(item.get("rerank_score", 0.0), 4),
            "rrf_score"       : round(item.get("rrf_score", 0.0), 6),
            "bm25_rank"       : item.get("bm25_rank"),
            "semantic_rank"   : item.get("semantic_rank"),

            "name"            : src.get("name"),
            "current_title"   : src.get("current_title"),
            "seniority"       : src.get("seniority"),
            "total_years_experience": src.get("total_years_experience"),
            "location"        : src.get("location", {}),
            "email"           : src.get("email"),

            "skills_flat"     : cand_skills,
            "matched_skills"  : matched,
            "missing_skills"  : missing,
            "match_pct"       : round(len(matched) / max(len(job_skills), 1) * 100, 1),

            "summary"         : src.get("summary"),
            "profile_quality_score": src.get("profile_quality_score"),
            "certifications"  : src.get("certifications") or [],
            "languages_spoken": src.get("languages_spoken") or [],
            "education"       : src.get("education") or [],
            "work_experience" : (src.get("work_experience") or [])[:3],
        })
    return results
