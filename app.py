# app.py
# PAD Job Creation Analyzer – rebuilt & fixed
# -------------------------------------------------------------
import re
from io import BytesIO
from typing import Tuple, Dict, List, Any, Optional

import altair as alt
import pandas as pd
import PyPDF2
import requests
import streamlit as st

# -------------------------------------------------------------
# Streamlit Config & Header
# -------------------------------------------------------------
st.set_page_config(page_title="PAD Job Analyzer", layout="wide", page_icon="📄")
st.title("PAD Job Creation Analyzer")
st.caption(
    "Upload a PAD PDF. The app will estimate direct/indirect jobs from the text, "
    "detect sector cues, and surface comparable World Bank projects."
)

# -------------------------------------------------------------
# Sidebar: Assumptions & Controls
# -------------------------------------------------------------
st.sidebar.header("Assumptions")
jobs_per_million: float = st.sidebar.number_input(
    "Jobs per US$1 million",
    min_value=0.0,
    max_value=5000.0,
    value=200.0,
    step=10.0,
    help="Average jobs supported per US$1 million invested (adjust per sector/context).",
)
direct_share_pct: float = st.sidebar.slider(
    "Direct share of jobs (%)",
    min_value=0,
    max_value=100,
    value=35,
    step=1,
    help="Percent of total jobs considered 'direct'; remainder is 'indirect'.",
)
direct_pct: float = direct_share_pct / 100.0

plus_minus_pct: int = st.sidebar.slider(
    "Uncertainty band (±%)", min_value=5, max_value=60, value=25, step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("“Better Jobs” shares when signals are present")
better_direct_share: float = st.sidebar.slider(
    "Better share of direct jobs (%)", 0, 100, 40, 5
) / 100.0
better_indirect_share: float = st.sidebar.slider(
    "Better share of indirect jobs (%)", 0, 100, 20, 5
) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Sector multipliers")
# You can tune these multipliers to reflect sector-specific job intensity
sector_multipliers: Dict[str, float] = {
    "Agriculture": 1.15,
    "Manufacturing": 1.25,
    "Energy": 0.85,
    "Transport": 0.9,
    "ICT": 0.8,
    "Health": 1.05,
    "Education": 1.0,
    "Finance / Private Sector": 0.95,
    "Water": 0.95,
    "Urban": 1.0,
    "Social Protection": 0.9,
    "Other / General": 1.0,
}
with st.sidebar.expander("View multipliers used"):
    st.json(sector_multipliers, expanded=False)

# -------------------------------------------------------------
# Text Utilities
# -------------------------------------------------------------
def clean_text(txt: str) -> str:
    """Normalize whitespace, de-hyphenate line breaks."""
    if not txt:
        return ""
    # Remove hyphenation at line ends (e.g., "employ-\nment" -> "employment")
    txt = re.sub(r"-\s*\n\s*", "", txt)
    # Replace newlines with space and collapse whitespace
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def extract_text_from_pdf(file) -> str:
    """
    Extract text using PyPDF2. Handles encrypted PDFs when possible.
    """
    try:
        reader = PyPDF2.PdfReader(file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # try empty password
            except Exception:
                pass
        pages_text = []
        for p in reader.pages:
            try:
                pages_text.append(p.extract_text() or "")
            except Exception:
                pages_text.append("")
        return "\n".join(pages_text)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""


# -------------------------------------------------------------
# Investment Amount Parsing (Million USD)
# -------------------------------------------------------------
INV_PATTERNS: List[re.Pattern] = [
    # US$ / USD / $ with unit
    re.compile(r"(?:US\$|USD|\$)\s*([\d,]+(?:\.\d+)?)\s*(million|billion)", re.I),
    # Unit then currency (sometimes)
    re.compile(r"([\d,]+(?:\.\d+)?)\s*(million|billion)\s*(?:US\$|USD|\$)?", re.I),
    # Phrases like "Total project cost ... US$ X million"
    re.compile(r"(?:total (?:project )?cost|financing(?: amount)?|loan amount)\D{0,30}(?:US\$|USD|\$)?\s*([\d,]+(?:\.\d+)?)\s*(million|billion)", re.I),
]

def parse_investment_amount_million(text: str) -> Tuple[Optional[float], str, str]:
    """
    Return (amount_in_million, confidence, snippet)
    """
    text = text or ""
    candidates: List[Tuple[float, str]] = []

    for pat in INV_PATTERNS:
        for m in pat.finditer(text):
            num = m.group(1)
            unit = m.group(2) if len(m.groups()) >= 2 else None
            if not num or not unit:
                continue
            try:
                val = float(num.replace(",", ""))
                if "billion" in unit.lower():
                    val *= 1000.0
                # Save value with matched snippet
                snippet = m.group(0)
                candidates.append((val, snippet))
            except Exception:
                continue

    if not candidates:
        # ultra-fallback: any "<number> million/billion"
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*(million|billion)", text, flags=re.I)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if m.group(2).lower() == "billion":
                    val *= 1000.0
                return val, "Medium (generic amount found)", m.group(0)
            except Exception:
                pass
        return None, "Low (no clear amount found, default used)", "No clear investment amount found."

    # Choose the max (PADs often highlight a total); reduce confidence if multiple distinct values exist
    best_val, best_snip = max(candidates, key=lambda x: x[0])
    distinct_vals = {round(v, 2) for v, _ in candidates}
    if len(distinct_vals) > 1:
        return best_val, "Medium (multiple amounts detected)", best_snip
    return best_val, "High", best_snip


# -------------------------------------------------------------
# Sector Detection
# -------------------------------------------------------------
SECTOR_CUES: Dict[str, List[str]] = {
    "Agriculture": ["agriculture", "agri-", "agro", "irrigation", "farm", "rural livelihoods"],
    "Manufacturing": ["manufactur", "factory", "industrial park", "industry 4.0"],
    "Energy": ["energy", "electric", "power", "generation", "grid", "renewable", "solar", "wind"],
    "Transport": ["transport", "road", "highway", "rail", "port", "logistics", "corridor"],
    "ICT": ["ict", "digital", "broadband", "connectivity", "data center", "e-government"],
    "Health": ["health", "clinic", "hospital", "public health", "disease"],
    "Education": ["education", "school", "tvet", "teacher", "learning", "curriculum"],
    "Finance / Private Sector": ["msme", "sme", "finance", "credit", "lending", "bank", "guarantee"],
    "Water": ["water", "sanitation", "wastewater", "wss", "utility"],
    "Urban": ["urban", "municipal", "city", "housing", "land use"],
    "Social Protection": ["social protection", "cash transfer", "safety net", "public works"],
}

def find_sentence_with_keyword(text: str, keyword: str) -> str:
    # Simple sentence split by period; robust enough for evidence display
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if keyword.lower() in sent.lower():
            return sent.strip()
    return ""

def detect_sector(text: str) -> Tuple[str, str]:
    """
    Return (sector, evidence_sentence)
    """
    txt = text.lower()
    scores: Dict[str, int] = {}
    evidence = ""
    for sector, cues in SECTOR_CUES.items():
        score = 0
        for cue in cues:
            score += txt.count(cue.lower())
        if score > 0 and not evidence:
            # get first evidence sentence for the first hit cue
            for cue in cues:
                ev = find_sentence_with_keyword(text, cue)
                if ev:
                    evidence = ev
                    break
        if score:
            scores[sector] = score

    if not scores:
        return "Other / General", ""

    best_sector = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_sector, evidence


# -------------------------------------------------------------
# Better Jobs Signals
# -------------------------------------------------------------
BETTER_JOB_KEYWORDS = {
    "skills", "training", "apprentice",
    "capacity building",
    "labor standards", "labour standards",
    "osh", "occupational safety",
    "formalization", "formalisation",
    "compliance", "decent work",
    "wage", "earnings", "productivity",
    "women", "female", "gender",
    "youth employment", "social dialogue",
    "working conditions",
}

def detect_better_jobs_signals(text: str) -> List[str]:
    txt = text.lower()
    hits = [k for k in BETTER_JOB_KEYWORDS if k in txt]
    return sorted(set(hits))


# -------------------------------------------------------------
# World Bank API Helpers
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def worldbank_projects_search(params: Dict[str, Any]) -> Dict[str, Any]:
    base_url = "https://search.worldbank.org/api/v2/projects"
    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def sector_query_value(sector: str) -> str:
    return sector if sector != "Other / General" else ""

def jobs_from_description_score(description: str) -> int:
    """
    Lightweight heuristic score when explicit job counts are not available.
    """
    keywords = ["employment", "job", "labor", "labour", "workforce", "hiring", "recruitment", "livelihood", "skills", "training"]
    score = sum(description.lower().count(k) for k in keywords)
    return score * 100

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_equivalent_projects(sector: str, net_commitment_million: float, tolerance: float = 0.25) -> pd.DataFrame:
    params = {
        "format": "json",
        "statuscode_exact": "C",  # Closed
        "rows": 2000,
    }
    if sector_query_value(sector):
        params["sectorname"] = sector_query_value(sector)

    try:
        data = worldbank_projects_search(params)
        projects = list(data.get("projects", {}).values())
    except Exception:
        projects = []

    rows = []
    for p in projects:
        try:
            commitment = float(str(p.get("totalcommamt", "0")).replace(",", "")) / 1_000_000.0
        except Exception:
            continue

        if net_commitment_million and net_commitment_million > 0:
            rel = abs(commitment - net_commitment_million) / max(net_commitment_million, 1e-9)
            if rel > tolerance:
                continue

        description = ""
        abs_ = p.get("project_abstract")
        if isinstance(abs_, dict):
            description = abs_.get("cdata", "") or ""
        elif isinstance(abs_, str):
            description = abs_

        rows.append(
            {
                "Project Name": p.get("project_name", "Unnamed"),
                "Country": p.get("countryshortname", "Unknown"),
                "P-Code": p.get("id", ""),
                "Dates": f"{(p.get('boardapprovaldate','') or '')[:10]} to {(p.get('closingdate','') or '')[:10]}",
                "Est. Jobs (text score)": jobs_from_description_score(description),
                "Total Commitment (US$M)": round(commitment, 2),
            }
        )
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_projects(sector: str, require_jobs: bool = True) -> pd.DataFrame:
    """
    Active projects; optionally filter to those that mention jobs in results framework.
    """
    params = {
        "format": "json",
        "statuscode_exact": "A",  # Active
        "rows": 2000,
    }
    if sector_query_value(sector):
        params["sectorname"] = sector_query_value(sector)

    try:
        data = worldbank_projects_search(params)
        projects = list(data.get("projects", {}).values())
    except Exception:
        projects = []

    rows = []
    for p in projects:
        rf = p.get("resultsframework", "") or ""
        jobs_mentioned: Optional[int] = None

        if isinstance(rf, str) and rf:
            # e.g., "12,345 jobs created" / "15,000 employment positions added"
            m = re.search(
                r"(\d[\d,]*)\s+(?:jobs?|employment|positions?)\s+(?:created|added|supported)",
                rf,
                flags=re.I,
            )
            if m:
                try:
                    jobs_mentioned = int(m.group(1).replace(",", ""))
                except Exception:
                    jobs_mentioned = None

        if require_jobs and jobs_mentioned is None:
            continue

        try:
            commitment = float(str(p.get("totalcommamt", "0")).replace(",", "")) / 1_000_000.0
        except Exception:
            commitment = 0.0

        rows.append(
            {
                "Project Name": p.get("project_name", "Unnamed"),
                "Country": p.get("countryshortname", "Unknown"),
                "P-Code": p.get("id", ""),
                "Dates": f"{(p.get('boardapprovaldate','') or '')[:10]} to {(p.get('closingdate','') or '')[:10]}",
                "Total Commitment (US$M)": round(commitment, 2),
                "Jobs Mentioned": jobs_mentioned if jobs_mentioned is not None else "N/A",
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------
# Core Estimation Logic
# -------------------------------------------------------------
def estimate_uncertainty(value: int, pct: int) -> Tuple[int, int]:
    lower = int(round(value * (1 - pct / 100)))
    upper = int(round(value * (1 + pct / 100)))
    return lower, upper

def estimate_jobs_logic(
    text: str,
    jobs_per_million: float,
    direct_pct: float,
    sector_multipliers: Dict[str, float],
    better_direct_share: float,
    better_indirect_share: float,
) -> Dict[str, Any]:
    text = text or ""

    sector, sector_sentence = detect_sector(text)
    amount_m, conf, amount_snip = parse_investment_amount_million(text)

    used_default = False
    if amount_m is None:
        amount_m = 50.0  # sensible default when not found
        used_default = True

    # Apply sector multiplier
    mult = sector_multipliers.get(sector, sector_multipliers["Other / General"])

    base_jobs = amount_m * jobs_per_million * mult
    direct_jobs = int(base_jobs * direct_pct)
    indirect_jobs = int(base_jobs * (1.0 - direct_pct))

    # “Better jobs” detection
    better_hits = detect_better_jobs_signals(text)
    better_flag = len(better_hits) > 0
    better_direct = int(direct_jobs * better_direct_share) if better_flag else 0
    better_indirect = int(indirect_jobs * better_indirect_share) if better_flag else 0

    # Evidence quote (one sentence mentioning jobs/skills/employment, etc.)
    quote_match = re.search(
        r"([^\.]*\b(job creation|employment|labor|labour|msmes?|skills|training|enterprise|firms)\b[^\.]*\.)",
        text,
        flags=re.I,
    )
    source_quote = quote_match.group(1).strip() if quote_match else "No specific jobs/skills sentence found."

    # Confidence assembly
    overall_conf = "High"
    if sector == "Other / General":
        overall_conf = "Medium"
    if used_default:
        overall_conf = "Low"

    return {
        "sector": sector,
        "investment_estimate_million_usd": float(round(amount_m, 2)),
        "amount_confidence": conf,
        "investment_snippet": amount_snip,
        "sector_evidence": sector_sentence,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "better_direct_jobs": better_direct,
        "better_indirect_jobs": better_indirect,
        "better_jobs_signals": better_hits,
        "confidence": overall_conf,
        "source_quote": source_quote,
    }


# -------------------------------------------------------------
# UI – File Upload & Results
# -------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PAD PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        full_text = extract_text_from_pdf(uploaded_file)
        full_text_clean = clean_text(full_text)

    with st.spinner("Estimating jobs..."):
        results = estimate_jobs_logic(
            full_text_clean,
            jobs_per_million=jobs_per_million,
            direct_pct=direct_pct,
            sector_multipliers=sector_multipliers,
            better_direct_share=better_direct_share,
            better_indirect_share=better_indirect_share,
        )

    st.subheader("Job Creation Estimate")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sector", results["sector"])
        st.caption(results["sector_evidence"] or "No sector evidence sentence found.")
    with col2:
        st.metric("Investment (US$M)", f"{results['investment_estimate_million_usd']:.2f}")
        st.caption(f"{results['amount_confidence']} — “{results['investment_snippet']}”")
    with col3:
        st.metric("Overall Confidence", results["confidence"])
        st.caption("Based on sector detection and investment parsing.")

    # Direct / Indirect with uncertainty bands
    dj_low, dj_high = estimate_uncertainty(results["direct_jobs"], plus_minus_pct)
    ij_low, ij_high = estimate_uncertainty(results["indirect_jobs"], plus_minus_pct)

    st.markdown("### Jobs Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Direct Jobs:** {results['direct_jobs']:,} (±{plus_minus_pct}% → {dj_low:,}–{dj_high:,})")
        st.markdown(f"**Better Direct Jobs:** {results['better_direct_jobs']:,}")
    with c2:
        st.markdown(f"**Indirect Jobs:** {results['indirect_jobs']:,} (±{plus_minus_pct}% → {ij_low:,}–{ij_high:,})")
        st.markdown(f"**Better Indirect Jobs:** {results['better_indirect_jobs']:,}")

    # Evidence
    st.subheader("Source Evidence")
    st.markdown(f"**Quoted Text:**\n> {results['source_quote']}")
    if results["better_jobs_signals"]:
        st.info("**Better Jobs Signals Detected:** " + ", ".join(results["better_jobs_signals"]))
    else:
        st.info("No better-jobs signals detected; set to 0 by current rules.")

    # Comparable closed projects
    st.subheader("Comparable Closed Projects (Same/Similar Sector & Size)")
    eq_df = fetch_equivalent_projects(
        results["sector"],
        results["investment_estimate_million_usd"],
        tolerance=0.25,
    )
    if not eq_df.empty:
        st.dataframe(eq_df, use_container_width=True)

        # Chart
        chart_data = eq_df[["Project Name", "Est. Jobs (text score)"]].copy()
        chart_data = chart_data[chart_data["Est. Jobs (text score)"] > 0]
        if not chart_data.empty:
            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("Est. Jobs (text score):Q", title="Estimated Jobs (heuristic score)"),
                    y=alt.Y("Project Name:N", sort='-x', title="Project"),
                    tooltip=["Project Name", "Est. Jobs (text score)"],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No comparable projects with text-based job signals.")
    else:
        st.info("No comparable closed projects found for the chosen filters.")

    # Live projects
    st.subheader("Active (Live) Projects in the Same Sector")
    st.caption("Filtered to those that mention jobs in the results framework; toggle to see all.")
    live_df = fetch_live_projects(results["sector"], require_jobs=True)
    if not live_df.empty:
        st.dataframe(live_df, use_container_width=True)
    else:
        st.info("No live projects found that mention jobs in the results framework.")

    show_all = st.checkbox("Show all live projects in this sector")
    if show_all:
        all_live_df = fetch_live_projects(results["sector"], require_jobs=False)
        if not all_live_df.empty:
            st.dataframe(all_live_df, use_container_width=True)
        else:
            st.warning("No live projects found at all for this sector.")

    # Download results
    st.subheader("Download Results")
    out = {
        **results,
        "assumptions_jobs_per_million": jobs_per_million,
        "assumptions_direct_share": direct_pct,
        "assumptions_indirect_share": 1.0 - direct_pct,
        "assumptions_sector_multiplier_used": sector_multipliers.get(results["sector"], 1.0),
        "uncertainty_pct": plus_minus_pct,
    }
    out_df = pd.DataFrame([out])

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Job Estimate")
        if not eq_df.empty:
            eq_df.to_excel(writer, index=False, sheet_name="Comparable Projects")
        if not live_df.empty:
            live_df.to_excel(writer, index=False, sheet_name="Live Projects (Jobs)")

    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="pad_job_estimate.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("Methodology & Notes"):
        st.markdown(
            "* **Estimation approach**\n"
            "  - Identify sector cues in PAD text and map to broad sectors for comparables.\n"
            "  - Parse investment amounts from multiple patterns (US$, USD, million/billion, financing cues).\n"
            "  - Compute base jobs = (US$M) × (jobs per US$1M), scaled by sector multiplier and direct/indirect shares.\n"
            "  - Detect 'better jobs' via PAD text signals (skills, OSH, standards, formalization, wages, women/youth, etc.).\n"
            "    Apply configurable shares to direct/indirect jobs when signals are present.\n"
            "  - Show ± uncertainty bands for direct/indirect totals.\n\n"
            "* **Comparable projects**\n"
            "  - Closed projects in the same sector and within ±25% of the investment amount.\n"
            "  - Where no explicit jobs are available, a text-signal heuristic is shown for context only.\n\n"
            "* **Limitations**\n"
            "  - PDF text extraction may miss tables; values could be under/over captured.\n"
            "  - Sector detection from text is approximate; consider manual override if needed.\n"
            "  - 'Better jobs' signals are proxies; for formal reporting, align with task-team agreed indicators.\n\n"
            "* **Tips**\n"
            "  - Tweak the sliders in the sidebar to run sensitivity analyses.\n"
            "  - If PADs follow a template, add a parser for \"Financing Table\" and \"Results Framework\" sections.\n"
        )
else:
    st.info("Upload a PAD PDF to begin.")
