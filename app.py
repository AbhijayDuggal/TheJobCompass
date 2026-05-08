"""
JobCompass — Streamlit Frontend
All UI code lives here. All logic is in src/.

Run: streamlit run app.py
"""

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st

# Add src/ to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobCompass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        color: white; padding: 2rem 2.5rem; border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.75rem 1rem; margin: 0.25rem 0;
    }
    .status-ok   { border-left: 4px solid #22c55e; }
    .status-warn { border-left: 4px solid #f59e0b; }
    .status-err  { border-left: 4px solid #ef4444; }
    .result-card {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1.25rem; margin: 0.75rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .skill-tag {
        display: inline-block; background: #dbeafe; color: #1e40af;
        border-radius: 4px; padding: 0.15rem 0.5rem;
        font-size: 0.8rem; margin: 0.1rem;
    }
    .skill-match  { background: #dcfce7; color: #166534; }
    .skill-miss   { background: #fee2e2; color: #991b1b; }
    .score-badge {
        display: inline-block; background: #1e3a5f; color: white;
        border-radius: 20px; padding: 0.2rem 0.7rem; font-weight: 600;
        font-size: 0.85rem;
    }
    .progress-log {
        background: #0f172a; color: #94a3b8; border-radius: 8px;
        padding: 1rem; font-family: monospace; font-size: 0.8rem;
        max-height: 250px; overflow-y: auto;
    }
    .quality-bar-outer {
        background: #e2e8f0; border-radius: 4px; height: 8px;
        margin: 0.3rem 0;
    }
    .missing-field-warning {
        background: #fef3c7; border: 1px solid #f59e0b;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ──────────────────────────────────────────────────────

def init_session():
    defaults = {
        "log_lines"       : [],
        "candidate_result": None,
        "recruiter_result": None,
        "current_candidate": None,
        "current_job"     : None,
        "system_status"   : None,
        "last_status_check": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_lines.append(f"[{ts}] {msg}")
    if len(st.session_state.log_lines) > 200:
        st.session_state.log_lines = st.session_state.log_lines[-200:]


def get_log_text() -> str:
    return "\n".join(st.session_state.log_lines[-60:])


# ── System status (cached 10s) ─────────────────────────────────────────────────

def fetch_status() -> dict:
    now = time.time()
    if now - st.session_state.last_status_check < 10 and st.session_state.system_status:
        return st.session_state.system_status
    try:
        from pipeline_orchestrator import get_system_status, get_vram_info
        s = get_system_status()
        s["vram"] = get_vram_info()
    except Exception as e:
        s = {
            "opensearch": False, "vllm_ocr": False, "vllm_text": False,
            "candidates": 0, "jobs": 0, "data_ready": False,
            "vram": {"available": False}, "error": str(e)
        }
    st.session_state.system_status = s
    st.session_state.last_status_check = now
    return s


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧭 JobCompass")
        st.caption("AI-Powered Job Matching")
        st.divider()

        status = fetch_status()

        st.markdown("**System Status**")

        def status_line(label, ok, detail=""):
            icon  = "🟢" if ok else "🔴"
            extra = f" `{detail}`" if detail else ""
            st.markdown(f"{icon} {label}{extra}")

        status_line("OpenSearch",   status["opensearch"])
        status_line("vLLM OCR",     status["vllm_ocr"],   "port 8000")
        status_line("vLLM Text",    status["vllm_text"],  "port 8001")

        vram = status.get("vram", {})
        if vram.get("available"):
            pct = (vram["used_gb"] / vram["total_gb"]) * 100
            st.markdown(f"🎮 GPU: `{vram['name'].split()[-1]}`")
            st.progress(pct / 100, text=f"{vram['used_gb']:.1f}/{vram['total_gb']}GB VRAM")
        else:
            st.markdown("💻 CPU mode (no GPU detected)")

        st.divider()
        st.markdown("**Index Stats**")
        st.metric("Candidates", f"{status['candidates']:,}")
        st.metric("Jobs",       f"{status['jobs']:,}")

        st.divider()
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.session_state.last_status_check = 0
            st.rerun()

        st.divider()
        st.markdown("**Server Commands**")
        with st.expander("OCR Server (port 8000)"):
            from pipeline_orchestrator import get_vllm_ocr_command
            st.code(get_vllm_ocr_command(), language="bash")
        with st.expander("Text Server (port 8001)"):
            from pipeline_orchestrator import get_vllm_text_command
            st.code(get_vllm_text_command(), language="bash")
        with st.expander("OpenSearch Docker"):
            st.code(
                "docker run -d --name opensearch -p 9200:9200 \\\n"
                "  -e discovery.type=single-node \\\n"
                "  -e DISABLE_SECURITY_PLUGIN=true \\\n"
                "  -e OPENSEARCH_JAVA_OPTS=\"-Xms2g -Xmx2g\" \\\n"
                "  opensearchproject/opensearch:2.13.0",
                language="bash"
            )


# ── Skill tags ─────────────────────────────────────────────────────────────────

def render_skills(skills: list, style: str = ""):
    cls = f"skill-tag {style}"
    tags = " ".join(f'<span class="{cls}">{s}</span>' for s in (skills or []))
    st.markdown(tags, unsafe_allow_html=True)


# ── Progress log box ───────────────────────────────────────────────────────────

def render_log(placeholder):
    placeholder.markdown(
        f'<div class="progress-log">{get_log_text().replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True
    )


# ── Candidate tab ──────────────────────────────────────────────────────────────

def render_candidate_tab():
    st.markdown("## 👤 Find Jobs Matching Your Profile")
    st.caption("Upload your resume or enter your Candidate ID to find the best job matches.")

    status = fetch_status()

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "How would you like to proceed?",
        ["Upload Resume (PDF)", "Enter Candidate ID"],
        horizontal=True,
    )

    col_input, col_filters = st.columns([2, 1])

    with col_input:
        if input_mode == "Upload Resume (PDF)":
            uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        else:
            candidate_id = st.text_input("Candidate ID", placeholder="e.g. 0042")
            uploaded = None

    with col_filters:
        st.markdown("**🔍 Filters**")
        seniority_opts = ["fresher", "junior", "mid", "senior", "lead", "executive"]
        filter_seniority = st.multiselect("Seniority level", seniority_opts,
                                          help="Filter by job seniority")
        filter_location  = st.text_input("City / Location", placeholder="e.g. New York",
                                         help="Filter by job location")
        filter_work_type = st.selectbox("Work type",
                                        ["Any", "Full-Time", "Part-Time", "Contract", "Remote", "Hybrid"],
                                        help="Filter by work arrangement")

    # Optional extra info for uploaded resumes
    extra_info = {}
    if input_mode == "Upload Resume (PDF)" and uploaded:
        with st.expander("📝 Fill in any missing details (improves matching)"):
            c1, c2, c3 = st.columns(3)
            extra_info["name"]  = c1.text_input("Full Name", key="cand_name")
            extra_info["email"] = c2.text_input("Email", key="cand_email")
            extra_info["phone"] = c3.text_input("Phone", key="cand_phone")
            c4, c5 = st.columns(2)
            extra_info["current_title"]  = c4.text_input("Current/Target Job Title", key="cand_title")
            extra_info["total_years_experience"] = c5.number_input("Years of Experience", min_value=0, max_value=50, value=0, key="cand_exp")
            extra_info["summary"] = st.text_area("Professional Summary (optional)", height=80, key="cand_summary")
            extra_info["skills_text"] = st.text_input("Key Skills (comma-separated)", key="cand_skills",
                                                       placeholder="Python, Machine Learning, Docker...")

    st.divider()
    find_btn = st.button("🔎 Find Matching Jobs", type="primary", use_container_width=True)
    log_ph   = st.empty()
    result_ph = st.container()

    if find_btn:
        if not status["data_ready"]:
            st.error("Index is empty. Run the full pipeline first from the Admin tab.")
            return

        work_type_val = None if filter_work_type == "Any" else filter_work_type

        st.session_state.log_lines = []
        render_log(log_ph)

        if input_mode == "Enter Candidate ID":
            if not candidate_id.strip():
                st.warning("Please enter a Candidate ID.")
                return
            with st.spinner("Fetching candidate and finding jobs..."):
                try:
                    log("Fetching candidate from index...")
                    render_log(log_ph)
                    from step4_indexing import get_candidate_by_id
                    candidate = get_candidate_by_id(candidate_id.strip())
                    if not candidate:
                        st.error(f"Candidate ID '{candidate_id}' not found in index.")
                        return
                    log(f"Found: {candidate.get('name')} | {candidate.get('current_title')}")
                    render_log(log_ph)
                    _run_job_search(candidate, filter_seniority, filter_location, work_type_val, log_ph)
                except Exception as e:
                    st.error(f"Error: {e}")
                    log(f"ERROR: {e}")
                    render_log(log_ph)

        else:  # Upload PDF
            if not uploaded:
                st.warning("Please upload a PDF resume.")
                return

            # if not (status["vllm_ocr"] and status["vllm_text"]):
            #     st.error(
            #         "Both vLLM servers must be running to process a new resume.\n\n"
            #         "Start them using the commands in the sidebar, then try again."
            #     )
            #     return

            with st.spinner("Processing resume... (this may take 1-2 minutes)"):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    cid = str(uuid.uuid4())[:8]
                    log(f"Processing PDF: {uploaded.name} -> temp ID: {cid}")
                    render_log(log_ph)

                    from pipeline_orchestrator import process_new_resume

                    resume_output = process_new_resume(
                        tmp_path,
                        cid,
                        lambda m: (log(m), render_log(log_ph))
                    )

                    candidate = resume_output["candidate"]
                    raw_record = resume_output["raw_record"]
                    
                    with st.expander("📄 Raw OCR Extracted Text", expanded=False):

                        st.markdown(f"### Candidate ID: `{candidate.get('candidate_id', cid)}`")

                        st.text_area(
                            "OCR Text",
                            raw_record.get("raw_text", ""),
                            height=400,
                            key=f"ocr_text_{cid}",
                        )

                    with st.expander("🧠 Structured Candidate Profile", expanded=True):

                        st.markdown(f"### Candidate ID: `{candidate.get('candidate_id', cid)}`")

                        st.json(candidate)

                    with st.expander("🔎 Candidate Embedding Text", expanded=False):

                        st.text_area(
                            "Embedding Text",
                            candidate.get("embedding_text", ""),
                            height=300,
                            key=f"candidate_embedding_{cid}",
                        )

                    # Merge extra info
                    for key, val in extra_info.items():
                        if val and key != "skills_text":
                            candidate[key] = val
                    if extra_info.get("skills_text"):
                        extra_skills = [s.strip() for s in extra_info["skills_text"].split(",") if s.strip()]
                        candidate["skills_flat"] = list(dict.fromkeys(
                            (candidate.get("skills_flat") or []) + extra_skills
                        ))

                    os.unlink(tmp_path)
                    with st.expander("📄 Raw OCR Extracted Text", expanded=False):

                        st.markdown(f"### Candidate ID: `{candidate.get('candidate_id', cid)}`")

                        st.text_area(
                            "OCR Text",
                            raw_record.get("raw_text", ""),
                            height=400,
                        )
                    st.session_state.current_candidate = candidate
                    _run_job_search(candidate, filter_seniority, filter_location, work_type_val, log_ph)

                except Exception as e:
                    st.error(f"Error processing resume: {e}")
                    log(f"ERROR: {e}")
                    render_log(log_ph)

    # Render last results
    if st.session_state.candidate_result:
        _render_job_results(
            st.session_state.candidate_result["jobs"],
            st.session_state.candidate_result["candidate"],
            result_ph,
        )


def _run_job_search(candidate, filter_seniority, filter_location, filter_work_type, log_ph):
    from pipeline_orchestrator import get_retrieval_engine

    # Quality gate
    quality = candidate.get("profile_quality_score", 0)
    missing = candidate.get("missing_fields") or []

    if quality < 50 and missing:
        st.warning(
            f"⚠️ **Profile quality score: {quality}/100** — results may be less accurate.\n\n"
            f"Missing information: **{', '.join(missing)}**\n\n"
            "Use the 'Fill in any missing details' section above to improve matching."
        )

    log("Getting retrieval engine...")
    render_log(log_ph)
    engine = get_retrieval_engine(lambda m: (log(m), render_log(log_ph)))

    log("Running hybrid retrieval...")
    render_log(log_ph)
    jobs = engine.find_jobs_for_candidate(
        candidate,
        filter_seniority=filter_seniority if filter_seniority else None,
        filter_location=filter_location.strip() if filter_location.strip() else None,
        filter_work_type=filter_work_type,
        progress_cb=lambda m: (log(m), render_log(log_ph)),
    )

    st.session_state.candidate_result = {"jobs": jobs, "candidate": candidate}
    log(f"Found {len(jobs)} matching jobs.")
    render_log(log_ph)


def _render_job_results(jobs: list, candidate: dict, container):
    with container:
        st.markdown(f"### 🎯 Top {len(jobs)} Job Matches")
        if candidate.get("name"):
            st.caption(f"Results for: **{candidate['name']}** | {candidate.get('current_title','')} | Quality: {candidate.get('profile_quality_score',0)}/100")

        for job in jobs:
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(
                        f"**#{job['rank']}  {job['title']}**  "
                        f"`[{job.get('job_id', 'NO_ID')}]`"
                    )
                    company  = job.get("company") or "—"
                    loc      = job.get("location", {})
                    loc_str  = ", ".join(filter(None, [loc.get("city"), loc.get("country")])) or "—"
                    work_type = job.get("work_type") or "—"
                    st.caption(f"🏢 {company}  📍 {loc_str}  🏷️ {work_type}  📊 {job.get('seniority','—')}")
                with h2:
                    st.markdown(f'<span class="score-badge">Score: {job["rerank_score"]:.3f}</span>', unsafe_allow_html=True)
                    match_pct = job.get("match_pct", 0)
                    st.progress(match_pct / 100, text=f"Skill match: {match_pct:.0f}%")

                salary = job.get("salary", {})
                if salary.get("min") or salary.get("max"):
                    lo = salary.get("min", "?")
                    hi = salary.get("max", "?")
                    st.caption(f"💰 Salary: ${lo:,} – ${hi:,}" if isinstance(lo, int) else f"💰 {salary.get('raw','')}")

                if job.get("responsibilities_summary"):
                    st.markdown(f"_{job['responsibilities_summary']}_")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**✅ Matched skills:**")
                    render_skills(job.get("matched_skills", []), "skill-match")
                with c2:
                    st.markdown("**❌ Missing skills:**")
                    render_skills(job.get("missing_skills", []), "skill-miss")

                if job.get("domain_tags"):
                    render_skills(job["domain_tags"])

                with st.expander("Show full description"):

                        st.write(job.get("description", "No description available."))

                        st.divider()

                        st.markdown("### ⚙ Retrieval Debug Info")

                        st.json({
                            "job_id": job.get("job_id"),
                            "bm25_rank": job.get("bm25_rank"),
                            "semantic_rank": job.get("semantic_rank"),
                            "rrf_score": job.get("rrf_score"),
                            "rerank_score": job.get("rerank_score"),
                        })

                st.markdown('</div>', unsafe_allow_html=True)


# ── Recruiter tab ──────────────────────────────────────────────────────────────

def render_recruiter_tab():
    st.markdown("## 🏢 Find Candidates for Your Job")
    st.caption("Upload a job description or enter a Job ID to find the best candidate matches.")

    status = fetch_status()

    input_mode = st.radio(
        "How would you like to proceed?",
        ["Upload/Paste Job Description", "Enter Job ID"],
        horizontal=True,
    )

    col_input, col_filters = st.columns([2, 1])

    with col_input:
        if input_mode == "Upload/Paste Job Description":
            jd_text = st.text_area(
                "Paste job description here",
                height=200,
                placeholder="Paste the full job description including title, responsibilities, requirements..."
            )
            col_a, col_b = st.columns(2)
            job_title   = col_a.text_input("Job Title *", placeholder="e.g. Senior Data Engineer")
            company_name = col_b.text_input("Company Name", placeholder="e.g. Acme Corp")
            col_c, col_d = st.columns(2)
            work_location = col_c.text_input("Location", placeholder="e.g. San Francisco")
            work_type_in  = col_d.selectbox("Work Type", ["Full-Time","Part-Time","Contract","Remote","Hybrid"])
            required_skills_raw = st.text_input(
                "Required Skills (comma-separated)",
                placeholder="Python, SQL, Spark, AWS..."
            )
            experience_req = st.text_input("Experience Required", placeholder="3-5 years")
        else:
            job_id_input = st.text_input("Job ID", placeholder="e.g. job_0042")
            jd_text = None

    with col_filters:
        st.markdown("**🔍 Candidate Filters**")
        seniority_opts = ["fresher", "junior", "mid", "senior", "lead", "executive"]
        filter_seniority = st.multiselect("Seniority level", seniority_opts)
        filter_location  = st.text_input("Candidate location", placeholder="e.g. New York")
        st.markdown("**⚙️ Result Settings**")
        top_n = st.slider("Max results", 5, 20, 10)

    st.divider()
    find_btn = st.button("🔎 Find Matching Candidates", type="primary", use_container_width=True)
    log_ph    = st.empty()
    result_ph = st.container()

    if find_btn:
        if not status["data_ready"]:
            st.error("Index is empty. Run the full pipeline first from the Admin tab.")
            return

        st.session_state.log_lines = []
        render_log(log_ph)

        if input_mode == "Enter Job ID":
            if not job_id_input.strip():
                st.warning("Please enter a Job ID.")
                return
            with st.spinner("Fetching job and finding candidates..."):
                try:
                    log(f"Fetching job {job_id_input.strip()}...")
                    render_log(log_ph)
                    from step4_indexing import get_job_by_id
                    job = get_job_by_id(job_id_input.strip())
                    if not job:
                        st.error(f"Job ID '{job_id_input.strip()}' not found in index.")
                        return
                    log(f"Found: {job.get('normalized_title')} at {job.get('company')}")
                    render_log(log_ph)
                    _run_candidate_search(job, filter_seniority, filter_location, top_n, log_ph)
                except Exception as e:
                    st.error(f"Error: {e}")
                    log(f"ERROR: {e}")
                    render_log(log_ph)

        else:  # Upload/Paste JD
            if not job_title.strip():
                st.warning("Please provide at least a Job Title.")
                return
            if not jd_text.strip():
                st.warning("Please paste the job description.")
                return

            if not (status["vllm_text"]):
                st.error("vLLM text server (port 8001) must be running to process a new job description.")
                return

            with st.spinner("Processing job description..."):
                try:
                    jid = f"job_{uuid.uuid4().hex[:8]}"
                    job_data = {
                        "Job Title"      : job_title,
                        "Job Description": jd_text,
                        "Responsibilities": "",
                        "skills"         : required_skills_raw,
                        "Company"        : company_name,
                        "location"       : work_location,
                        "Work Type"      : work_type_in,
                        "Experience"     : experience_req,
                    }
                    log(f"Processing JD: {job_title} [{jid}]")
                    render_log(log_ph)
                    from pipeline_orchestrator import process_new_job
                    from pipeline_orchestrator import process_new_job

                    job_output = process_new_job(
                        job_data,
                        jid,
                        lambda m: (log(m), render_log(log_ph))
                    )

                    job = job_output["job"]
                    with st.expander("💼 Structured Job Data", expanded=True):

                        st.markdown(f"### Job ID: `{job.get('job_id', jid)}`")

                        st.json(job)
                    with st.expander("🔎 Job Embedding Text", expanded=False):

                        st.text_area(
                            "Embedding Text",
                            job.get("embedding_text", ""),
                            height=300,
                            key=f"job_embedding_{jid}",
                        )    
                    
                    st.session_state.current_job = job
                    _run_candidate_search(job, filter_seniority, filter_location, top_n, log_ph)
                except Exception as e:
                    st.error(f"Error processing job description: {e}")
                    log(f"ERROR: {e}")
                    render_log(log_ph)

    if st.session_state.recruiter_result:
        _render_candidate_results(
            st.session_state.recruiter_result["candidates"],
            st.session_state.recruiter_result["job"],
            result_ph,
        )


def _run_candidate_search(job, filter_seniority, filter_location, top_n, log_ph):
    from pipeline_orchestrator import get_retrieval_engine
    log("Getting retrieval engine...")
    render_log(log_ph)
    engine = get_retrieval_engine(lambda m: (log(m), render_log(log_ph)))
    engine.rerank_top_n = top_n

    log("Running hybrid retrieval for candidates...")
    render_log(log_ph)
    candidates = engine.find_candidates_for_job(
        job,
        filter_seniority=filter_seniority if filter_seniority else None,
        filter_location=filter_location.strip() if filter_location.strip() else None,
        progress_cb=lambda m: (log(m), render_log(log_ph)),
    )
    st.session_state.recruiter_result = {"candidates": candidates, "job": job}
    log(f"Found {len(candidates)} matching candidates.")
    render_log(log_ph)


def _render_candidate_results(candidates: list, job: dict, container):
    with container:
        job_title = job.get("normalized_title") or job.get("title", "")
        st.markdown(f"### 🎯 Top {len(candidates)} Candidates for: {job_title}")
        if job.get("company"):
            st.caption(f"🏢 {job['company']}  |  Skills: {', '.join((job.get('required_skills') or [])[:6])}")

        for c in candidates:
            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                h1, h2 = st.columns([3, 1])
                with h1:
                    name  = c.get("name") or f"Candidate {c['candidate_id']}"
                    title = c.get("current_title") or "—"
                    st.markdown(
                        f"**#{c['rank']}  {name}**  "
                        f"`[{c.get('candidate_id', 'NO_ID')}]`"
                    )
                    exp   = c.get("total_years_experience")
                    exp_s = f"{exp}y exp" if exp else "—"
                    loc   = c.get("location", {})
                    loc_s = loc.get("city") or loc.get("country") or "—"
                    st.caption(f"💼 {title}  📍 {loc_s}  ⏱️ {exp_s}  📊 {c.get('seniority','—')}")
                    if c.get("email"):
                        st.caption(f"📧 {c['email']}")
                with h2:
                    st.markdown(f'<span class="score-badge">Score: {c["rerank_score"]:.3f}</span>', unsafe_allow_html=True)
                    match_pct = c.get("match_pct", 0)
                    q_score   = c.get("profile_quality_score", 0)
                    st.progress(match_pct / 100, text=f"Skill match: {match_pct:.0f}%")
                    st.caption(f"Profile quality: {q_score}/100")

                if c.get("summary"):
                    st.markdown(f"_{c['summary'][:200]}_")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matched skills:**")
                    render_skills(c.get("matched_skills", []), "skill-match")
                with col2:
                    st.markdown("**❌ Missing skills:**")
                    render_skills(c.get("missing_skills", []), "skill-miss")

                st.markdown("**All skills:**")
                render_skills((c.get("skills_flat") or [])[:15])

                with st.expander("Show education & experience"):
                    for edu in (c.get("education") or []):
                        deg  = edu.get("degree","")
                        inst = edu.get("institution","")
                        yr   = edu.get("graduation_year","")
                        st.caption(f"🎓 {deg} — {inst} ({yr})")
                    for exp_item in (c.get("work_experience") or []):
                        ttl = exp_item.get("title","")
                        co  = exp_item.get("company","")
                        desc = exp_item.get("description","")
                        st.caption(f"💼 {ttl} @ {co}: {desc}")
                    st.divider()

                    st.markdown("### ⚙ Retrieval Debug Info")

                    st.json({
                        "candidate_id": c.get("candidate_id"),
                        "bm25_rank": c.get("bm25_rank"),
                        "semantic_rank": c.get("semantic_rank"),
                        "rrf_score": c.get("rrf_score"),
                        "rerank_score": c.get("rerank_score"),
                    })

                st.markdown('</div>', unsafe_allow_html=True)


# ── Admin / Pipeline tab ───────────────────────────────────────────────────────

def render_admin_tab():
    st.markdown("## ⚙️ Admin — Pipeline Management")
    st.caption("Run the full data pipeline to build the index from scratch.")

    status = fetch_status()

    st.markdown("### 📋 Prerequisites Checklist")
    c1, c2, c3 = st.columns(3)
    c1.metric("OpenSearch", "✅ Ready" if status["opensearch"] else "❌ Not running")
    c2.metric("vLLM OCR (8000)", "✅ Ready" if status["vllm_ocr"] else "❌ Not running")
    c3.metric("vLLM Text (8001)", "✅ Ready" if status["vllm_text"] else "❌ Not running")

    st.divider()

    st.markdown("### 🚀 Full Pipeline")
    st.info(
        "Steps:\n"
        "1. **Resume Extraction** — OCR all PDFs using vLLM (requires port 8000)\n"
        "2. **Profile & Job Extraction** — LLM structured extraction (requires port 8001)\n"
        "3. **Embedding Generation** — Qwen3-Embedding-0.6B (GPU, auto load/unload)\n"
        "4. **OpenSearch Indexing** — BM25 + HNSW kNN index"
    )

    all_servers_ok = status["opensearch"] and status["vllm_ocr"] and status["vllm_text"]
    if not all_servers_ok:
        st.warning("⚠️ Not all servers are running. Some steps may fail. Start them using the commands in the sidebar.")

    col_run, col_force = st.columns([3, 1])
    force_reindex = col_force.checkbox("Force reindex (deletes existing data)", value=False)

    log_ph = st.empty()

    if col_run.button("▶️ Run Full Pipeline", type="primary", use_container_width=True):
        st.session_state.log_lines = []
        render_log(log_ph)
        with st.spinner("Running pipeline... check log below for progress"):
            try:
                from pipeline_orchestrator import run_full_pipeline
                run_full_pipeline(lambda m: (log(m), render_log(log_ph)))
                st.success("✅ Pipeline complete!")
                st.session_state.last_status_check = 0
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                log(f"FATAL: {e}")
                render_log(log_ph)

    render_log(log_ph)

    st.divider()
    st.markdown("### 🔧 Individual Steps")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown("**Step 1: OCR**")
        st.caption("Extract text from PDFs")
        if st.button("Run Step 1", key="s1", use_container_width=True):
            _run_step_button("step1", log_ph)

    with s2:
        st.markdown("**Step 2: Extract**")
        st.caption("LLM structured extraction")
        if st.button("Run Step 2", key="s2", use_container_width=True):
            _run_step_button("step2", log_ph)

    with s3:
        st.markdown("**Step 3: Embed**")
        st.caption("Generate vectors")
        if st.button("Run Step 3", key="s3", use_container_width=True):
            _run_step_button("step3", log_ph)

    with s4:
        st.markdown("**Step 4: Index**")
        st.caption("Load into OpenSearch")
        if st.button("Run Step 4", key="s4", use_container_width=True):
            _run_step_button("step4", log_ph)

    st.divider()
    st.markdown("### 💾 VRAM Management")
    col_a, col_b = st.columns(2)
    if col_a.button("🗑️ Unload retrieval models from VRAM", use_container_width=True):
        try:
            from pipeline_orchestrator import unload_retrieval_models
            unload_retrieval_models(log)
            st.success("Models unloaded from VRAM.")
        except Exception as e:
            st.error(str(e))
    if col_b.button("📊 Show VRAM usage", use_container_width=True):
        from pipeline_orchestrator import get_vram_info
        info = get_vram_info()
        if info["available"]:
            st.info(f"GPU: {info['name']} | Total: {info['total_gb']}GB | Used: {info['used_gb']}GB | Free: {info['free_gb']}GB")
        else:
            st.info("No GPU detected.")


def _run_step_button(step: str, log_ph):
    st.session_state.log_lines = []
    render_log(log_ph)
    try:
        if step == "step1":
            from step1_resume_extract import run
            run(lambda m: (log(m), render_log(log_ph)))
        elif step == "step2":
            from step2_profile_extraction import run
            run(lambda m: (log(m), render_log(log_ph)))
        elif step == "step3":
            from step3_embedding import run
            run(lambda m: (log(m), render_log(log_ph)))
        elif step == "step4":
            from step4_indexing import run
            run(lambda m: (log(m), render_log(log_ph)))
        st.success(f"Step {step[-1]} complete.")
        st.session_state.last_status_check = 0
    except Exception as e:
        st.error(f"Step failed: {e}")
        log(f"ERROR: {e}")
        render_log(log_ph)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()

    st.markdown(
        '<div class="main-header">'
        '<h1>🧭 JobCompass</h1>'
        '<p style="margin:0;opacity:0.85">AI-powered job matching using hybrid BM25 + semantic retrieval + cross-encoder reranking</p>'
        '</div>',
        unsafe_allow_html=True
    )

    tab_candidate, tab_recruiter, tab_admin = st.tabs([
        "👤 Candidate — Find Jobs",
        "🏢 Recruiter — Find Candidates",
        "⚙️ Admin — Pipeline",
    ])

    with tab_candidate:
        render_candidate_tab()

    with tab_recruiter:
        render_recruiter_tab()

    with tab_admin:
        render_admin_tab()


if __name__ == "__main__":
    main()
