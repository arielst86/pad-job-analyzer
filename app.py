# appold2.py
# PAD Job Creation Analyzer — Evidence-first, no assumption toggles

import re
from io import BytesIO
from typing import Tuple, Dict, List, Any, Optional
import streamlit as st
import pandas as pd
import PyPDF2

# -----------------------------
# Streamlit Config & Header
# -----------------------------
st.set_page_config(page_title="PAD Job Analyzer", layout="wide", page_icon="📄")
st.title("PAD Job Creation Analyzer (Evidence‑First)")
st.caption(
    "Upload a PAD PDF. The app will extract **explicit** direct and indirect job counts from the PAD text "
    "and quote the **exact** sentences (with page numbers) used to derive those counts."
)

# -----------------------------
# Text / PDF Utilities
# -----------------------------
def clean_text_basic(txt: str) -> str:
    """
    Light cleanup: de-hyphenate linebreaks and collapse whitespace.
    """
    if not txt:
        return ""
    txt = re.sub(r"-\s*\n\s*", "", txt)      # e.g., employ-\nment -> employment
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def extract_text_per_page(file) -> List[str]:
    """
    Returns a list of strings: one cleaned text chunk per page.
    Also resilient to encrypted PDFs (attempts empty password).
    """
    pages = []
    try:
        reader = PyPDF2.PdfReader(file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # try empty password
            except Exception:
                pass
        for p in reader.pages:
            try:
                raw = p.extract_text() or ""
            except Exception:
                raw = ""
            pages.append(clean_text_basic(raw))
    except Exception as e:
        st.error(f"PDF read error: {e}")
    return pages

def sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Find sentence boundaries given a match span within text.
    A 'good enough' approach using ., ?, ! as delimiters.
    """
    left = text.rfind('.', 0, start)
    ql = text.rfind('?', 0, start)
    el = text.rfind('!', 0, start)
    left = max(left, ql, el)

    right_dot = text.find('.', end)
    right_q = text.find('?', end)
    right_e = text.find('!', end)
    candidates = [c for c in [right_dot, right_q, right_e] if c != -1]
    right = min(candidates) if candidates else len(text) - 1

    # Expand to include the delimiter char
    if right < len(text) - 1:
        right += 1
    # Clip lower bound to 0 if not found
    if left == -1:
        left = 0
    else:
        # Move after the delimiter
        left += 1
    return (left, right)

def exact_sentence(text: str, span: Tuple[int, int]) -> str:
    a, b = sentence_bounds(text, span[0], span[1])
    return text[a:b].strip()

def to_int(num_str: str) -> Optional[int]:
    try:
        s = num_str.replace(",", "").strip()
        # handle decimals but cast down to int since PAD counts are integers
        return int(float(s))
    except Exception:
        return None

# -----------------------------
# Job Extraction Patterns
# -----------------------------
# Reject units that are NOT simple job counts
NON_JOB_UNITS = r"(?:person[-\s]?days?|man[-\s]?days?|worker[-\s]?days?|job[-\s]?years?|job[-\s]?months?)"

# Paired patterns in a single sentence
PAIR_PATTERNS = [
    # "<direct> direct jobs ... and <indirect> indirect jobs"
    re.compile(
        rf"(?P<direct>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)\s+(?:and|&|,)\s+(?P<indirect>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)",
        re.I),
    # "<indirect> indirect jobs ... and <direct> direct jobs"
    re.compile(
        rf"(?P<indirect>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)\s+(?:and|&|,)\s+(?P<direct>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)",
        re.I),
    # "Z jobs, of which X direct and Y indirect"
    re.compile(
        rf"(?P<total>\d[\d,\.]*)\s+(?:jobs?|employment)\b[^\.]*?\bof which\b[^\.]*?(?P<direct>\d[\d,\.]*)\s+direct[^\.]*?(?:and|,)\s+(?P<indirect>\d[\d,\.]*)\s+indirect",
        re.I),
]

# Single mentions
DIRECT_PATTERNS = [
    re.compile(rf"(?P<num>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)\b", re.I),
    re.compile(rf"(?:jobs?|employment|FTEs?)\s+direct(?:ly)?\s*(?P<num>\d[\d,\.]*)\b", re.I),
]
INDIRECT_PATTERNS = [
    re.compile(rf"(?P<num>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)\b", re.I),
]
TOTAL_PATTERNS = [
    re.compile(rf"(?P<num>\d[\d,\.]*)\s+(?:jobs?|employment)\s+(?:created|generated|supported|expected|targets?)\b", re.I),
    re.compile(rf"total\s+(?:jobs?|employment)\s*(?:=|:)?\s*(?P<num>\d[\d,\.]*)\b", re.I),
]

# Exclusion: if NON_JOB_UNITS near number, ignore that match
NEAR_EXCLUSION = re.compile(NON_JOB_UNITS, re.I)

# Section ranking (Results Framework > PDO > Economic Analysis > anything else)
SECTION_RANKS = [
    (re.compile(r"results framework|annex\s+1[:\s-]*\s*results", re.I), 0),
    (re.compile(r"pdo indicators?|project development objective", re.I), 1),
    (re.compile(r"economic analysis|annex\s+4[:\s-]*\s*economic", re.I), 2),
]
DEFAULT_RANK = 3

def page_rank(pages: List[str], page_index: int) -> int:
    """
    Assign a rank based on section headings on the same page (best) or nearby pages.
    """
    text = pages[page_index]
    for rx, rnk in SECTION_RANKS:
        if rx.search(text):
            return rnk
    # Look within +/- 1 page for a nearby heading
    for offset in (-1, 1):
        j = page_index + offset
        if 0 <= j < len(pages):
            near = pages[j]
            for rx, rnk in SECTION_RANKS:
                if rx.search(near):
                    return rnk + 0.5  # slightly worse than on-page
    return DEFAULT_RANK

# -----------------------------
# Core Extraction
# -----------------------------
def find_job_mentions(pages: List[str]) -> Dict[str, Any]:
    """
    Traverse pages and return best evidence-backed job counts.
    Returns keys:
        direct, indirect, total            (ints or None)
        direct_evidence, indirect_evidence, total_evidence  (dicts with page, quote) if used
        method, confidence                 (strings)
    """
    pairs: List[Dict[str, Any]] = []
    singles_direct: List[Dict[str, Any]] = []
    singles_indirect: List[Dict[str, Any]] = []
    totals: List[Dict[str, Any]] = []

    for i, text in enumerate(pages):
        if not text:
            continue

        # Ignore hits that are clearly not job counts (e.g., person-days) by local check around matches later.
        # 1) Pairs
        for rx in PAIR_PATTERNS:
            for m in rx.finditer(text):
                span = m.span()
                sent = exact_sentence(text, span)
                if NEAR_EXCLUSION.search(sent):
                    continue
                d = to_int(m.groupdict().get("direct") or "")
                ind = to_int(m.groupdict().get("indirect") or "")
                tot = to_int(m.groupdict().get("total") or "")
                if d is None and ind is None:
                    continue
                rank = page_rank(pages, i)
                pairs.append({
                    "direct": d,
                    "indirect": ind,
                    "total": tot,
                    "page": i + 1,  # human-friendly 1-based
                    "quote": sent,
                    "rank": rank
                })

        # 2) Singles
        for rx in DIRECT_PATTERNS:
            for m in rx.finditer(text):
                span = m.span()
                sent = exact_sentence(text, span)
                if NEAR_EXCLUSION.search(sent):
                    continue
                val = to_int(m.group("num"))
                if val is None:
                    continue
                singles_direct.append({
                    "num": val,
                    "page": i + 1,
                    "quote": sent,
                    "rank": page_rank(pages, i)
                })

        for rx in INDIRECT_PATTERNS:
            for m in rx.finditer(text):
                span = m.span()
                sent = exact_sentence(text, span)
                if NEAR_EXCLUSION.search(sent):
                    continue
                val = to_int(m.group("num"))
                if val is None:
                    continue
                singles_indirect.append({
                    "num": val,
                    "page": i + 1,
                    "quote": sent,
                    "rank": page_rank(pages, i)
                })

        for rx in TOTAL_PATTERNS:
            for m in rx.finditer(text):
                span = m.span()
                sent = exact_sentence(text, span)
                if NEAR_EXCLUSION.search(sent):
                    continue
                val = to_int(m.group("num"))
                if val is None:
                    continue
                totals.append({
                    "num": val,
                    "page": i + 1,
                    "quote": sent,
                    "rank": page_rank(pages, i)
                })

    # Decision logic
    # A) Best explicit pair
    if pairs:
        # Sort by rank then prefer Results Framework (rank 0) and near-pair statements
        pairs_sorted = sorted(pairs, key=lambda x: (x["rank"], x["page"]))
        best = pairs_sorted[0]
        return {
            "direct": best["direct"],
            "indirect": best["indirect"],
            "total": best["total"],
            "direct_evidence": {"page": best["page"], "quote": best["quote"]} if best["direct"] is not None else None,
            "indirect_evidence": {"page": best["page"], "quote": best["quote"]} if best["indirect"] is not None else None,
            "total_evidence": {"page": best["page"], "quote": best["quote"]} if best["total"] is not None else None,
            "method": "Explicit pair in one sentence",
            "confidence": "High"
        }

    # B) Combine singles on the same or adjacent page
    if singles_direct and singles_indirect:
        # Rank by (rank, |page distance|); prefer same page and better sections
        candidates = []
        for d in singles_direct:
            for ind in singles_indirect:
                dist = abs(d["page"] - ind["page"])
                if dist <= 1:
                    candidates.append((max(d["rank"], ind["rank"]), dist, d, ind))
        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], min(t[2]["page"], t[3]["page"])))
            _, _, d, ind = candidates[0]
            return {
                "direct": d["num"],
                "indirect": ind["num"],
                "total": None,
                "direct_evidence": {"page": d["page"], "quote": d["quote"]},
                "indirect_evidence": {"page": ind["page"], "quote": ind["quote"]},
                "total_evidence": None,
                "method": "Combined separate direct & indirect mentions (same/adjacent page)",
                "confidence": "Medium"
            }

    # C) Derive from explicit total
    # If we have total + direct -> indirect; or total + indirect -> direct
    if totals:
        totals_sorted = sorted(totals, key=lambda x: (x["rank"], x["page"]))
        best_total = totals_sorted[0]
        # Try total + direct
        if singles_direct:
            # Prefer direct on same/adjacent page
            with_d = sorted(singles_direct, key=lambda d: (abs(d["page"] - best_total["page"]), d["rank"]))
            for d in with_d:
                if abs(d["page"] - best_total["page"]) <= 1:
                    derived_ind = best_total["num"] - d["num"]
                    if derived_ind >= 0:
                        return {
                            "direct": d["num"],
                            "indirect": derived_ind,
                            "total": best_total["num"],
                            "direct_evidence": {"page": d["page"], "quote": d["quote"]},
                            "indirect_evidence": {
                                "page": best_total["page"],
                                "quote": f"Derived as Total – Direct using: “{best_total['quote']}” and “{d['quote']}”"
                            },
                            "total_evidence": {"page": best_total["page"], "quote": best_total["quote"]},
                            "method": "Derived from explicit total and direct",
                            "confidence": "Medium"
                        }
        # Try total + indirect
        if singles_indirect:
            with_i = sorted(singles_indirect, key=lambda i: (abs(i["page"] - best_total["page"]), i["rank"]))
            for ind in with_i:
                if abs(ind["page"] - best_total["page"]) <= 1:
                    derived_dir = best_total["num"] - ind["num"]
                    if derived_dir >= 0:
                        return {
                            "direct": derived_dir,
                            "indirect": ind["num"],
                            "total": best_total["num"],
                            "direct_evidence": {
                                "page": best_total["page"],
                                "quote": f"Derived as Total – Indirect using: “{best_total['quote']}” and “{ind['quote']}”"
                            },
                            "indirect_evidence": {"page": ind["page"], "quote": ind["quote"]},
                            "total_evidence": {"page": best_total["page"], "quote": best_total["quote"]},
                            "method": "Derived from explicit total and indirect",
                            "confidence": "Medium"
                        }

    # D) If multiple ambiguous singles far apart, pick best-ranked but flag low confidence
    if singles_direct or singles_indirect:
        # Choose best-ranked single(s) if only one side is available
        best_d = sorted(singles_direct, key=lambda d: (d["rank"], d["page"]))[0] if singles_direct else None
        best_i = sorted(singles_indirect, key=lambda i: (i["rank"], i["page"]))[0] if singles_indirect else None
        return {
            "direct": best_d["num"] if best_d else None,
            "indirect": best_i["num"] if best_i else None,
            "total": None,
            "direct_evidence": {"page": best_d["page"], "quote": best_d["quote"]} if best_d else None,
            "indirect_evidence": {"page": best_i["page"], "quote": best_i["quote"]} if best_i else None,
            "total_evidence": None,
            "method": "Single-sided mention(s) only (no reliable pairing/derivation)",
            "confidence": "Low"
        }

    # Nothing useful found
    return {
        "direct": None,
        "indirect": None,
        "total": None,
        "direct_evidence": None,
        "indirect_evidence": None,
        "total_evidence": None,
        "method": "No explicit jobs data found in PAD",
        "confidence": "N/A"
    }

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload a PAD PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        pages = extract_text_per_page(uploaded_file)

    if not any(pages):
        st.error("Could not extract any text from the PDF. It may be image-only or scanned.")
        st.stop()

    with st.spinner("Reading PAD for explicit job counts..."):
        results = find_job_mentions(pages)

    st.subheader("Job Creation (as explicitly stated in PAD)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Direct Jobs", f"{results['direct']:,}" if results['direct'] is not None else "Not stated in PAD")
    with c2:
        st.metric("Indirect Jobs", f"{results['indirect']:,}" if results['indirect'] is not None else "Not stated in PAD")
    with c3:
        st.metric("Confidence", results["confidence"])

    st.caption(f"Method: {results['method']}")

    # Evidence
    st.subheader("Exact Source Quotes (from the PAD)")
    if results["direct_evidence"]:
        st.markdown(f"**Direct jobs evidence** — page {results['direct_evidence']['page']}")
        st.markdown(f"> {results['direct_evidence']['quote']}")
    else:
        st.markdown("**Direct jobs evidence** — _None found in PAD_")

    if results["indirect_evidence"]:
        st.markdown(f"**Indirect jobs evidence** — page {results['indirect_evidence']['page']}")
        st.markdown(f"> {results['indirect_evidence']['quote']}")
    else:
        st.markdown("**Indirect jobs evidence** — _None found in PAD_")

    if results["total_evidence"]:
        st.markdown(f"**Total jobs evidence** — page {results['total_evidence']['page']}")
        st.markdown(f"> {results['total_evidence']['quote']}")

    # Download
    st.subheader("Download Results")
    out = {
        "direct_jobs": results["direct"],
        "indirect_jobs": results["indirect"],
        "total_jobs": results["total"],
        "confidence": results["confidence"],
        "method": results["method"],
        "direct_evidence_page": results["direct_evidence"]["page"] if results["direct_evidence"] else None,
        "direct_evidence_quote": results["direct_evidence"]["quote"] if results["direct_evidence"] else None,
        "indirect_evidence_page": results["indirect_evidence"]["page"] if results["indirect_evidence"] else None,
        "indirect_evidence_quote": results["indirect_evidence"]["quote"] if results["indirect_evidence"] else None,
        "total_evidence_page": results["total_evidence"]["page"] if results["total_evidence"] else None,
        "total_evidence_quote": results["total_evidence"]["quote"] if results["total_evidence"] else None,
    }
    df = pd.DataFrame([out])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Jobs from PAD")
    st.download_button(
        "Download as Excel",
        data=buffer.getvalue(),
        file_name="pad_jobs_explicit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("Upload a PAD PDF to begin.")
