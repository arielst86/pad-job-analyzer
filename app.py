import streamlit as st
import PyPDF2
import re
import pandas as pd
import requests
import altair as alt
from io import BytesIO
from typing import Tuple, Dict, List, Any, Optional

# -----------------------------
# Streamlit Config & Header
# -----------------------------
st.set_page_config(page_title="PAD Job Analyzer", layout="wide", page_icon="📄")

st.markdown(
    """
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
        <div style="font-size:2.0em;font-weight:700;color:#003366;">PAD Job Creation Analyzer</div>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/World_Bank_logo.svg/512px-World_Bank_logo.svg.png"
             unsafe_allow_html=True
)

# -----------------------------
# Sidebar Controls (Sensitivity)
# -----------------------------
with st.sidebar:
    st.header("Assumptions & Sensitivity")
    jobs_per_million = st.slider("Base jobs per US$1M", min_value=1, max_value=100, value=10, step=1)
    direct_pct = st.slider("Direct jobs share", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
    indirect_pct = 1.0 - direct_pct
    st.caption(f"Indirect share = {indirect_pct:.2f}")

    st.divider()
    st.subheader("Sector Multipliers")
    sector_multipliers = {
        "Education": st.slider("Education", 0.3, 3.0, 1.2, 0.1),
        "Health": st.slider("Health", 0.3, 3.0, 1.3, 0.1),
        "Agriculture and Food": st.slider("Agriculture and Food", 0.3, 3.0, 1.4, 0.1),
        "Transport": st.slider("Transport", 0.3, 3.0, 1.5, 0.1),
        "Energy & Extractives": st.slider("Energy & Extractives", 0.3, 3.0, 1.4, 0.1),
        "Digital Development": st.slider("Digital Development", 0.3, 3.0, 1.1, 0.1),
        "Other / General": st.slider("Other / General", 0.3, 3.0, 1.0, 0.1),
    }

    st.divider()
    st.subheader("Better Jobs Settings")
    better_direct_share = st.slider("Share of direct jobs considered 'better'", 0.0, 0.8, 0.30, 0.05)
    better_indirect_share = st.slider("Share of indirect jobs considered 'better'", 0.0, 0.8, 0.20, 0.05)

    st.divider()
    st.subheader("Estimate Uncertainty")
    plus_minus_pct = st.slider("Result ± uncertainty (%)", 0, 100, 20, 5)

# -----------------------------
# Helpers
# -----------------------------

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                pages.append(t)
        except Exception:
            continue
    return "\n".join(pages)

# Map free-text hits to WB-like sector names we can query
SECTOR_MAP = {
    "education": "Education",
    "school": "Education",
    "skills": "Education",
    "health": "Health",
    "hospital": "Health",
    "agriculture": "Agriculture and Food",
    "agricultural": "Agriculture and Food",
    "rural": "Agriculture and Food",
    "transport": "Transport",
    "road": "Transport",
    "highway": "Transport",
    "rail": "Transport",
    "energy": "Energy & Extractives",
    "power": "Energy & Extractives",
    "electricity": "Energy & Extractives",
    "mining": "Energy & Extractives",
    "digital": "Digital Development",
    "ict": "Digital Development",
    "connectivity": "Digital Development",
}

def detect_sector(text: str) -> Tuple[str, str]:
    """
    Return (sector_name_for_api, evidence_sentence)
    Falls back to 'Other / General' if none matched.
    """
    txt = text.lower()
    for k, v in SECTOR_MAP.items():
        if k in txt:
            # capture evidence sentence
            m = re.search(rf"([^.]*\b{k}\b[^.]*)\.", text, flags=re.IGNORECASE)
            return v, (m.group(1) + ".") if m else ""
    return "Other / General", ""

INV_PATTERNS = [
    # US$ 120 million / USD 120 million / $120 million
    r"(US\$|USD|\$)\s?([\d,]+(?:\.\d+)?)\s*(million|billion)",
    # 120 million USD
    r"([\d,]+(?:\.\d+)?)\s*(million|billion)\s*(US\$|USD|\$)",
    # 120 million (no currency)
    r"(?:amount|financing|cost|project cost|total financing|loan)\D{0,20}([\d,]+(?:\.\d+)?)\s*(million|billion)",
]

def parse_investment_amount_million(text: str) -> Tuple[Optional[float], str, str]:
    """
    Return (amount_in_million, confidence, snippet)
    """
    candidates = []
    for pat in INV_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            num = m.group(2) if len(m.groups()) >= 2 else None
            unit = m.group(3) if len(m.groups()) >= 3 else None
            if num and unit:
                try:
                    val = float(num.replace(",", ""))
                    unit = unit.lower()
                    if "billion" in unit:
                        val *= 1000.0
                    candidates.append((val, m.group(0)))
                except Exception:
                    pass

    if not candidates:
        # fallback to any "<number> million" anywhere
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*(million|billion)", text, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if m.group(2).lower() == "billion":
                    val *= 1000.0
                return val, "Medium (generic amount found)", m.group(0)
            except Exception:
                pass
        return None, "Low (no clear amount found, default used)", "No clear investment amount found."

    # Choose the most frequent or the largest (PAD often highlights total)
    # Heuristic: take the max; show snippet.
    best = max(candidates, key=lambda x: x[0])
    # If multiple distinct values exist, reduce confidence
    distinct_vals = {round(v, 2) for v, _ in candidates}
    if len(distinct_vals) > 1:
        return best[0], "Medium (multiple amounts detected)", best[1]
    return best[0], "High", best[1]

BETTER_JOB_KEYWORDS = {
    "skills", "training", "apprentice", "capacity building",
    "labor standards", "labour standards", "osh", "occupational safety",
    "formalization", "formalisation", "compliance", "decent work",
    "wage", "earnings", "productivity", "women", "female", "gender",
    "youth employment", "social dialogue", "working conditions"
}

def detect_better_jobs_signals(text: str) -> List[str]:
    txt = text.lower()
    hits = [k for k in BETTER_JOB_KEYWORDS if k in txt]
    # Deduplicate stems (e.g., wage/earnings both present)
    return sorted(set(hits))

@st.cache_data(show_spinner=False, ttl=3600)
def worldbank_projects_search(params: Dict[str, Any]) -> Dict[str, Any]:
    base_url = "https://search.worldbank.org/api/v2/projects"
    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def estimate_jobs_logic(
    text: str,
    jobs_per_million: float,
    direct_pct: float,
    sector_multipliers: Dict[str, float],
    better_direct_share: float,
    better_indirect_share: float
) -> Dict[str, Any]:
    text = text or ""
    sector, sector_sentence = detect_sector(text)

    amount_m, conf, amount_snip = parse_investment_amount_million(text)
    used_default = False
    if amount_m is None:
        amount_m = 50.0
        used_default = True

    # Apply sector multiplier
    mult = sector_multipliers.get(sector, sector_multipliers["Other / General"])

    base_jobs = amount_m * jobs_per_million
    direct_jobs = int(base_jobs * direct_pct * mult)
    indirect_jobs = int(base_jobs * (1.0 - direct_pct) * mult)

    # better jobs detection
    better_hits = detect_better_jobs_signals(text)
    better_flag = len(better_hits) > 0
    better_direct = int(direct_jobs * better_direct_share) if better_flag else 0
    better_indirect = int(indirect_jobs * better_indirect_share) if better_flag else 0

    # Pull a relevant evidence quote
    quote_match = re.search(
        r"([^.]*\b(job creation|employment|labor|labour|msmes|skills|training|enterprise|firms)\b[^.]*\.)",
        text, flags=re.IGNORECASE
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
        "source_quote": source_quote
    }

def estimate_uncertainty(value: int, pct: int) -> Tuple[int, int]:
    lower = int(round(value * (1 - pct / 100)))
    upper = int(round(value * (1 + pct / 100)))
    return lower, upper

def sector_query_value(sector: str) -> str:
    return sector if sector != "Other / General" else ""

def jobs_from_description_score(description: str) -> int:
    """Keep your lightweight heuristic for charting similar projects when no explicit jobs numbers exist."""
    keywords = ["employment", "job", "labor", "labour", "workforce", "hiring", "recruitment", "livelihood", "skills", "training"]
    score = sum(description.lower().count(k) for k in keywords)
    return score * 100

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_equivalent_projects(sector: str, net_commitment_million: float, tolerance: float = 0.25) -> pd.DataFrame:
    params = {
        "format": "json",
        "statuscode_exact": "C",   # Closed
        "rows": 2000
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
            rel = abs(commitment - net_commitment_million) / net_commitment_million
            if rel > tolerance:
                continue

        description = ""
        abs_ = p.get("project_abstract")
        if isinstance(abs_, dict):
            description = abs_.get("cdata", "") or ""
        elif isinstance(abs_, str):
            description = abs_

        rows.append({
            "Project Name": p.get("project_name", "Unnamed"),
            "Country": p.get("countryshortname", "Unknown"),
            "P-Code": p.get("id", ""),
            "Dates": f"{(p.get('boardapprovaldate','') or '')[:10]} to {(p.get('closingdate','') or '')[:10]}",
            "Est. Jobs (text score)": jobs_from_description_score(description),
            "Total Commitment (US$M)": round(commitment, 2)
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_projects(sector: str, require_jobs: bool = True) -> pd.DataFrame:
    """
    Active projects; optionally filter to those that mention jobs in results framework.
    """
    params = {
        "format": "json",
        "statuscode_exact": "A",
        "rows": 2000
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
        jobs_mentioned = None
        if isinstance(rf, str) and rf:
            m = re.search(r"(\d[\d,]*)\s+(?:job[s]?|employment|position[s]?|created|added)", rf, flags=re.IGNORECASE)
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

        rows.append({
            "Project Name": p.get("project_name", "Unnamed"),
            "Country": p.get("countryshortname", "Unknown"),
            "P-Code": p.get("id", ""),
            "Dates": f"{(p.get('boardapprovaldate','') or '')[:10]} to {(p.get('closingdate','') or '')[:10]}",
            "Total Commitment (US$M)": round(commitment, 2),
            "Jobs Mentioned": jobs_mentioned if jobs_mentioned is not None else "N/A"
        })

    return pd.DataFrame(rows)

# -----------------------------
# File Uploader
# -----------------------------
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
            better_indirect_share=better_indirect_share
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
        st.metric("Confidence", results["confidence"])
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

    # Equivalent (closed) projects
    st.subheader("Comparable Closed Projects (Same/Similar Sector & Size)")
    eq_df = fetch_equivalent_projects(results["sector"], results["investment_estimate_million_usd"], tolerance=0.25)
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
                    tooltip=["Project Name", "Est. Jobs (text score)"]
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
    st.caption("Filters to those that mention jobs in the results framework; toggle to see all.")
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
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

   with st.expander("Methodology & Notes"):
    md = (
        "**Estimation approach**\n"
        "- Identify sector cues in PAD text and map to WB-like sectors for comparables.\n"
        "- Parse investment amounts from multiple patterns (US$, USD, million/billion, Financing cues).\n"
        "- Compute base jobs = (US$M) x (jobs per US$1M), scaled by sector multiplier and direct/indirect shares.\n"
        "- Detect 'better jobs' via PAD text signals (skills, OSH, standards, formalization, wages, women/youth, etc.).\n"
        "  Apply configurable shares to direct/indirect jobs when signals are present.\n"
        "- Show +/- uncertainty bands for direct/indirect totals.\n\n"
        "**Comparable projects**\n"
        "- Closed projects in the same sector and within +/-25% of the investment amount.\n"
        "- Where no explicit jobs are available, a text-signal heuristic is shown for context only.\n\n"
        "**Limitations**\n"
        "- PDF text extraction may miss tables; values could be under/over captured.\n"
        "- Sector detection from text is approximate; consider manual override if needed.\n"
        "- 'Better jobs' signals are proxies; for formal reporting, align with task-team agreed indicators.\n\n"
        "**Tips**\n"
        "- Tweak the sliders in the sidebar to run sensitivity analyses.\n"
        "- If PADs follow a template, add a parser for \"Financing Table\" and \"Results Framework\" sections.\n"
    )
    st.markdown(md)
        """
    )        
else:
    st.info("Upload a PAD PDF to begin.")
