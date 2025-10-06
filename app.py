# appold2.py
# PAD Job Creation Analyzer — Explicit-first, AI fallback with traceable assumptions (no user toggles)

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
st.title("PAD Job Creation Analyzer")
st.caption(
    "The app (1) extracts **explicit** direct/indirect job counts from the PAD where stated, "
    "and (2) if absent, **estimates** them via AI based on PAD text (sector cues, financing, and signals). "
    "All numbers are tied to **exact PAD quotes** with page numbers."
)

# -----------------------------
# PDF & Text Helpers
# -----------------------------
def clean_text_basic(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"-\s*\n\s*", "", txt)  # de-hyphenate line breaks
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def extract_text_per_page(file) -> List[str]:
    pages = []
    try:
        reader = PyPDF2.PdfReader(file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # attempt empty password
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
    left = max(text.rfind(".", 0, start), text.rfind("?", 0, start), text.rfind("!", 0, start))
    right_candidates = [text.find(".", end), text.find("?", end), text.find("!", end)]
    right_candidates = [c for c in right_candidates if c != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    if left == -1:
        left = 0
    else:
        left += 1
    return (left, right)

def exact_sentence(text: str, span: Tuple[int, int]) -> str:
    a, b = sentence_bounds(text, span[0], span[1])
    return text[a:b].strip()

def to_int(num_str: str) -> Optional[int]:
    try:
        s = num_str.replace(",", "").strip()
        return int(float(s))
    except Exception:
        return None

# -----------------------------
# Investment Amount Parsing (US$ million)
# -----------------------------
INV_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:US\$|USD|\$)\s*([0-9][\d,]*(?:\.\d+)?)\s*(million|billion)", re.I),
    re.compile(r"([0-9][\d,]*(?:\.\d+)?)\s*(million|billion)\s*(?:US\$|USD|\$)?", re.I),
    re.compile(r"(?:total (?:project )?cost|financing(?: amount)?|loan amount)\D{0,30}(?:US\$|USD|\$)?\s*([0-9][\d,]*(?:\.\d+)?)\s*(million|billion)", re.I),
]

def parse_investment_amount_million(text: str) -> Tuple[Optional[float], str, Dict[str, Any]]:
    """
    Returns (amount_in_million, confidence_label, evidence_dict)
    evidence_dict = { 'quote': ..., 'page': ... } when available (page added by caller).
    This function returns text-level snippet; caller should map to page when possible.
    """
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
                snippet = m.group(0)
                candidates.append((val, snippet))
            except Exception:
                continue

    if not candidates:
        m = re.search(r"([0-9][\d,]*(?:\.\d+)?)\s*(million|billion)", text, flags=re.I)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if m.group(2).lower() == "billion":
                    val *= 1000.0
                return val, "Medium (generic amount found)", {"quote": m.group(0), "page": None}
            except Exception:
                pass
        return None, "Low (no clear amount found)", {"quote": "No clear financing amount detected in text.", "page": None}

    best_val, best_snip = max(candidates, key=lambda x: x[0])
    distinct_vals = {round(v, 2) for v, _ in candidates}
    conf = "High" if len(distinct_vals) == 1 else "Medium (multiple amounts detected)"
    return best_val, conf, {"quote": best_snip, "page": None}

# -----------------------------
# Explicit Job Extraction
# -----------------------------
NON_JOB_UNITS = r"(?:person[-\s]?days?|man[-\s]?days?|worker[-\s]?days?|job[-\s]?years?|job[-\s]?months?)"
NEAR_EXCLUSION = re.compile(NON_JOB_UNITS, re.I)

PAIR_PATTERNS = [
    re.compile(
        rf"(?P<direct>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)\s+(?:and|&|,)\s+(?P<indirect>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)",
        re.I),
    re.compile(
        rf"(?P<indirect>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)\s+(?:and|&|,)\s+(?P<direct>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)",
        re.I),
    re.compile(
        rf"(?P<total>\d[\d,\.]*)\s+(?:jobs?|employment)\b[^\.]*?\bof which\b[^\.]*?(?P<direct>\d[\d,\.]*)\s+direct[^\.]*?(?:and|,)\s+(?P<indirect>\d[\d,\.]*)\s+indirect",
        re.I),
]

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

SECTION_RANKS = [
    (re.compile(r"results framework|annex\s+1[:\s-]*\s*results", re.I), 0),
    (re.compile(r"pdo indicators?|project development objective", re.I), 1),
    (re.compile(r"economic analysis|annex\s+4[:\s-]*\s*economic", re.I), 2),
]
DEFAULT_RANK = 3

def page_rank(pages: List[str], page_index: int) -> float:
    page_text = pages[page_index]
    for rx, rnk in SECTION_RANKS:
        if rx.search(page_text):
            return float(rnk)
    for offset in (-1, 1):
        j = page_index + offset
        if 0 <= j < len(pages):
            near = pages[j]
            for rx, rnk in SECTION_RANKS:
                if rx.search(near):
                    return rnk + 0.5
    return float(DEFAULT_RANK)

def find_explicit_jobs(pages: List[str]) -> Optional[Dict[str, Any]]:
    pairs = []
    singles_direct, singles_indirect, totals = [], [], []

    for i, text in enumerate(pages):
        if not text:
            continue

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
                pairs.append({
                    "direct": d,
                    "indirect": ind,
                    "total": tot,
                    "page": i + 1,
                    "quote": sent,
                    "rank": page_rank(pages, i)
                })

        for rx in DIRECT_PATTERNS:
            for m in rx.finditer(text):
                sent = exact_sentence(text, m.span())
                if NEAR_EXCLUSION.search(sent):
                    continue
                val = to_int(m.group("num"))
                if val is None:
                    continue
                singles_direct.append({"num": val, "page": i + 1, "quote": sent, "rank": page_rank(pages, i)})

        for rx in INDIRECT_PATTERNS:
            for m in rx.finditer(text):
                sent = exact_sentence(text, m.span())
                if NEAR_EXCLUSION.search(sent):
                    continue
                val = to_int(m.group("num"))
                if val is None:
                    continue
                singles_indirect.append({"num": val, "page": i + 1, "quote": sent, "rank": page_rank(pages, i)})

        for rx in TOTAL_PATTERNS:
            for m in rx.finditer(text):
                sent = exact_sentence(text, m.span())
                if NEAR_EXCLUSION.search(sent):
                    continue
                val = to_int(m.group("num"))
                if val is None:
                    continue
                totals.append({"num": val, "page": i + 1, "quote": sent, "rank": page_rank(pages, i)})

    if pairs:
        pairs_sorted = sorted(pairs, key=lambda x: (x["rank"], x["page"]))
        best = pairs_sorted[0]
        return {
            "mode": "PAD-explicit",
            "direct": best["direct"],
            "indirect": best["indirect"],
            "total": best["total"],
            "confidence": "High",
            "method": "Explicit pair in one sentence",
            "evidence": {
                "jobs_quote": {"page": best["page"], "quote": best["quote"]}
            }
        }

    # Combine singles (same/adjacent page)
    if singles_direct and singles_indirect:
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
                "mode": "PAD-explicit",
                "direct": d["num"],
                "indirect": ind["num"],
                "total": None,
                "confidence": "Medium",
                "method": "Combined separate direct & indirect mentions (same/adjacent page)",
                "evidence": {
                    "direct_quote": {"page": d["page"], "quote": d["quote"]},
                    "indirect_quote": {"page": ind["page"], "quote": ind["quote"]},
                }
            }

    # Derive from explicit total + single
    if totals:
        totals_sorted = sorted(totals, key=lambda x: (x["rank"], x["page"]))
        best_total = totals_sorted[0]
        # total + direct
        if len(singles_direct) > 0:
            with_d = sorted(singles_direct, key=lambda d: (abs(d["page"] - best_total["page"]), d["rank"]))
            for d in with_d:
                if abs(d["page"] - best_total["page"]) <= 1:
                    derived_ind = best_total["num"] - d["num"]
                    if derived_ind >= 0:
                        return {
                            "mode": "PAD-explicit",
                            "direct": d["num"],
                            "indirect": derived_ind,
                            "total": best_total["num"],
                            "confidence": "Medium",
                            "method": "Derived from explicit total and direct",
                            "evidence": {
                                "total_quote": {"page": best_total["page"], "quote": best_total["quote"]},
                                "direct_quote": {"page": d["page"], "quote": d["quote"]},
                                "derivation": "Indirect = Total – Direct (see quotes)"
                            }
                        }
        # total + indirect
        if len(singles_indirect) > 0:
            with_i = sorted(singles_indirect, key=lambda i: (abs(i["page"] - best_total["page"]), i["rank"]))
            for ind in with_i:
                if abs(ind["page"] - best_total["page"]) <= 1:
                    derived_dir = best_total["num"] - ind["num"]
                    if derived_dir >= 0:
                        return {
                            "mode": "PAD-explicit",
                            "direct": derived_dir,
                            "indirect": ind["num"],
                            "total": best_total["num"],
                            "confidence": "Medium",
                            "method": "Derived from explicit total and indirect",
                            "evidence": {
                                "total_quote": {"page": best_total["page"], "quote": best_total["quote"]},
                                "indirect_quote": {"page": ind["page"], "quote": ind["quote"]},
                                "derivation": "Direct = Total – Indirect (see quotes)"
                            }
                        }

    return None  # no explicit data

# -----------------------------
# AI Fallback (no user toggles)
# -----------------------------
SECTOR_CUES: Dict[str, List[str]] = {
    "Agriculture": ["agriculture", "agri-", "agro", "irrigation", "farm", "rural livelihoods"],
    "Manufacturing": ["manufactur", "factory", "industrial park", "industry 4.0"],
    "Energy": ["energy", "electric", "power", "generation", "grid", "renewable", "solar", "wind", "hydro"],
    "Transport": ["transport", "road", "highway", "rail", "port", "logistics", "corridor"],
    "ICT": ["ict", "digital", "broadband", "connectivity", "data center", "e-government"],
    "Health": ["health", "clinic", "hospital", "public health", "disease"],
    "Education": ["education", "school", "tvet", "teacher", "learning", "curriculum"],
    "Finance / Private Sector": ["msme", "sme", "finance", "credit", "lending", "bank", "guarantee", "matching grant", "credit line"],
    "Water": ["water", "sanitation", "wastewater", "wss", "utility"],
    "Urban": ["urban", "municipal", "city", "housing", "land use"],
    "Social Protection": ["social protection", "cash transfer", "safety net", "public works", "cash-for-work", "labor-intensive"],
    "Other / General": []
}

AI_PRIORS: Dict[str, Dict[str, float]] = {
    # jobs_per_million_total = direct+indirect; direct_share = fraction of total
    "Agriculture": {"jobs_per_million_total": 180.0, "direct_share": 0.35},
    "Manufacturing": {"jobs_per_million_total": 220.0, "direct_share": 0.40},
    "Energy": {"jobs_per_million_total": 70.0, "direct_share": 0.30},
    "Transport": {"jobs_per_million_total": 90.0, "direct_share": 0.35},
    "ICT": {"jobs_per_million_total": 80.0, "direct_share": 0.35},
    "Health": {"jobs_per_million_total": 110.0, "direct_share": 0.35},
    "Education": {"jobs_per_million_total": 100.0, "direct_share": 0.35},
    "Finance / Private Sector": {"jobs_per_million_total": 140.0, "direct_share": 0.25},
    "Water": {"jobs_per_million_total": 95.0, "direct_share": 0.30},
    "Urban": {"jobs_per_million_total": 120.0, "direct_share": 0.35},
    "Social Protection": {"jobs_per_million_total": 200.0, "direct_share": 0.45},
    "Other / General": {"jobs_per_million_total": 120.0, "direct_share": 0.35},
}

ADJUSTMENT_RULES: List[Dict[str, Any]] = [
    # Each rule: pattern -> multiplier on jobs_per_million_total OR direct_share tweak
    {
        "name": "Labor-intensive / public works",
        "pattern": re.compile(r"\b(labor[-\s]?intensive|public works|cash[-\s]?for[-\s]?work)\b", re.I),
        "jobs_per_million_mult": 1.5,
        "direct_share_delta": +0.05
    },
    {
        "name": "MSME finance / credit line",
        "pattern": re.compile(r"\b(msme|sme|credit line|matching grant|partial credit guarantee)\b", re.I),
        "jobs_per_million_mult": 1.2,
        "direct_share_delta": -0.10
    },
    {
        "name": "Renewable generation focus",
        "pattern": re.compile(r"\b(solar|photovoltaic|wind|hydro|geothermal)\b", re.I),
        "jobs_per_million_mult": 0.9,
        "direct_share_delta": -0.02
    },
]

def detect_sector(pages: List[str]) -> Tuple[str, Dict[str, Any]]:
    joined = "\n".join(pages).lower()
    scores: Dict[str, int] = {}
    best_evidence = {"page": None, "quote": ""}
    for i, page in enumerate(pages):
        low = page.lower()
        for sector, cues in SECTOR_CUES.items():
            for cue in cues:
                if cue and cue in low:
                    scores[sector] = scores.get(sector, 0) + low.count(cue)
                    # capture first evidence
                    if best_evidence["quote"] == "":
                        m = re.search(re.escape(cue), low, re.I)
                        if m:
                            best_evidence = {"page": i + 1, "quote": exact_sentence(page, m.span())}
    if not scores:
        return "Other / General", best_evidence
    best_sector = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_sector, best_evidence

def map_amount_to_page_evidence(pages: List[str], snippet: str) -> Optional[Dict[str, Any]]:
    if not snippet:
        return None
    # Find the page where the snippet occurs
    for i, page in enumerate(pages):
        if snippet in page:
            return {"page": i + 1, "quote": snippet}
    # Fall back: look for numerically normalized snippet start
    return None

def detect_adjustments(pages: List[str]) -> List[Dict[str, Any]]:
    hits = []
    for i, page in enumerate(pages):
        for rule in ADJUSTMENT_RULES:
            for m in rule["pattern"].finditer(page):
                sent = exact_sentence(page, m.span())
                hits.append({
                    "name": rule["name"],
                    "jobs_per_million_mult": rule["jobs_per_million_mult"],
                    "direct_share_delta": rule["direct_share_delta"],
                    "page": i + 1,
                    "quote": sent
                })
    # Deduplicate by sentence and rule name
    seen = set()
    uniq = []
    for h in hits:
        key = (h["name"], h["quote"], h["page"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq

def compute_uncertainty(base_pct: int, amount_conf: str, sector_evidence_present: bool, n_rules: int) -> int:
    pct = base_pct
    if amount_conf.startswith("High"):
        pct -= 5
    elif amount_conf.startswith("Medium"):
        pct += 5
    else:  # Low
        pct += 10
    if sector_evidence_present:
        pct -= 2
    else:
        pct += 5
    if n_rules >= 2:
        pct -= 3  # more corroboration
    elif n_rules == 0:
        pct += 5  # less corroboration
    return max(10, min(50, pct))

def ai_estimate_jobs(pages: List[str]) -> Dict[str, Any]:
    full_text = "\n".join(pages)

    # Sector detection (with evidence)
    sector, sector_ev = detect_sector(pages)
    sector_ev_present = bool(sector_ev["quote"])

    # Financing amount parsing (scan full text; then try to locate page)
    amount_m, amount_conf, amount_ev = parse_investment_amount_million(full_text)
    if amount_ev and not amount_ev.get("page"):
        mapped = map_amount_to_page_evidence(pages, amount_ev.get("quote", ""))
        if mapped:
            amount_ev["page"] = mapped["page"]
            amount_ev["quote"] = mapped["quote"]

    used_default_amount = False
    if amount_m is None:
        amount_m = 50.0  # default when PAD amount is not detectable
        used_default_amount = True
        amount_conf = "Low (used default US$50M because financing not detected in PAD text)"
        amount_ev = {"page": None, "quote": "No clear financing amount found; default used for AI estimate."}

    # Initialize priors
    pri = AI_PRIORS.get(sector, AI_PRIORS["Other / General"])
    jobs_per_million = pri["jobs_per_million_total"]
    direct_share = pri["direct_share"]

    # Adjustment rules based on PAD signals (with quotes)
    adj_hits = detect_adjustments(pages)
    for h in adj_hits:
        jobs_per_million *= h["jobs_per_million_mult"]
        direct_share = max(0.05, min(0.9, direct_share + h["direct_share_delta"]))

    # Compute totals
    total_jobs = amount_m * jobs_per_million
    direct_jobs = int(round(total_jobs * direct_share))
    indirect_jobs = int(round(total_jobs - direct_jobs))

    # Uncertainty band based on evidence quality
    plus_minus_pct = compute_uncertainty(
        base_pct=25,
        amount_conf=amount_conf,
        sector_evidence_present=sector_ev_present,
        n_rules=len(adj_hits)
    )

    return {
        "mode": "AI-fallback",
        "sector": sector,
        "investment_musd": round(amount_m, 2),
        "assumptions": {
            "jobs_per_million_total": round(jobs_per_million, 2),
            "direct_share": round(direct_share, 3),
            "adjustment_rules_fired": [
                {
                    "name": h["name"],
                    "jobs_per_million_mult": h["jobs_per_million_mult"],
                    "direct_share_delta": h["direct_share_delta"],
                    "page": h["page"],
                    "quote": h["quote"],
                } for h in adj_hits
            ]
        },
        "direct": direct_jobs,
        "indirect": indirect_jobs,
        "uncertainty_pct": plus_minus_pct,
        "confidence": "Medium" if not used_default_amount else "Low",
        "evidence": {
            "sector_quote": sector_ev if sector_ev_present else None,
            "amount_quote": amount_ev,
            "note": "AI estimate derived from PAD sector cues and financing; see adjustment quotes."
        }
    }

# -----------------------------
# UI Flow
# -----------------------------
uploaded_file = st.file_uploader("Upload a PAD PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        pages = extract_text_per_page(uploaded_file)

    if not any(pages):
        st.error("No selectable text extracted. The PDF may be scanned or image-only. Consider OCR.")
        st.stop()

    # 1) Try explicit PAD extraction first
    explicit = find_explicit_jobs(pages)

    if explicit:
        # ----------------- EXPLICIT MODE -----------------
        st.subheader("Job Creation (stated explicitly in PAD)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Direct Jobs", f"{explicit['direct']:,}" if explicit['direct'] is not None else "—")
        with c2:
            st.metric("Indirect Jobs", f"{explicit['indirect']:,}" if explicit['indirect'] is not None else "—")
        with c3:
            st.metric("Total Jobs", f"{explicit['total']:,}" if explicit['total'] is not None else "—")
        with c4:
            st.metric("Confidence", explicit["confidence"])
        st.caption(f"Method: {explicit['method']} • Mode: {explicit['mode']}")

        st.subheader("Exact PAD Quotes")
        ev = explicit.get("evidence", {})
        if "jobs_quote" in ev and ev["jobs_quote"]:
            st.markdown(f"**Jobs statement** — page {ev['jobs_quote']['page']}")
            st.markdown(f"> {ev['jobs_quote']['quote']}")
        if "direct_quote" in ev and ev["direct_quote"]:
            st.markdown(f"**Direct** — page {ev['direct_quote']['page']}")
            st.markdown(f"> {ev['direct_quote']['quote']}")
        if "indirect_quote" in ev and ev["indirect_quote"]:
            st.markdown(f"**Indirect** — page {ev['indirect_quote']['page']}")
            st.markdown(f"> {ev['indirect_quote']['quote']}")
        if "total_quote" in ev and ev["total_quote"]:
            st.markdown(f"**Total** — page {ev['total_quote']['page']}")
            st.markdown(f"> {ev['total_quote']['quote']}")
        if "derivation" in ev and ev["derivation"]:
            st.info(ev["derivation"])

        # Download
        out = {
            "mode": explicit["mode"],
            "direct_jobs": explicit["direct"],
            "indirect_jobs": explicit["indirect"],
            "total_jobs": explicit["total"],
            "confidence": explicit["confidence"],
            "method": explicit["method"],
            "jobs_evidence_page": ev.get("jobs_quote", {}).get("page") if ev.get("jobs_quote") else None,
            "jobs_evidence_quote": ev.get("jobs_quote", {}).get("quote") if ev.get("jobs_quote") else None,
            "direct_evidence_page": ev.get("direct_quote", {}).get("page") if ev.get("direct_quote") else None,
            "direct_evidence_quote": ev.get("direct_quote", {}).get("quote") if ev.get("direct_quote") else None,
            "indirect_evidence_page": ev.get("indirect_quote", {}).get("page") if ev.get("indirect_quote") else None,
            "indirect_evidence_quote": ev.get("indirect_quote", {}).get("quote") if ev.get("indirect_quote") else None,
            "total_evidence_page": ev.get("total_quote", {}).get("page") if ev.get("total_quote") else None,
            "total_evidence_quote": ev.get("total_quote", {}).get("quote") if ev.get("total_quote") else None,
        }
        df = pd.DataFrame([out])
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="PAD-explicit")
        st.download_button(
            "Download as Excel",
            data=buf.getvalue(),
            file_name="pad_jobs_explicit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        # ----------------- AI FALLBACK MODE -----------------
        ai = ai_estimate_jobs(pages)

        st.subheader("Job Creation (AI fallback estimate)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Direct Jobs (est.)", f"{ai['direct']:,}")
        with c2:
            st.metric("Indirect Jobs (est.)", f"{ai['indirect']:,}")
        with c3:
            st.metric("Uncertainty (±)", f"{ai['uncertainty_pct']}%")
        with c4:
            st.metric("Confidence", ai["confidence"])
        st.caption(f"Mode: {ai['mode']} • Sector: {ai['sector']} • Investment used: US${ai['investment_musd']:.2f}M")

        # Evidence & Assumptions
        st.subheader("PAD Evidence that Drove the Estimate")
        if ai["evidence"].get("amount_quote"):
            aq = ai["evidence"]["amount_quote"]
            if aq.get("page"):
                st.markdown(f"**Financing amount** — page {aq['page']}")
            else:
                st.markdown("**Financing amount**")
            st.markdown(f"> {aq['quote']}")
        if ai["evidence"].get("sector_quote"):
            sq = ai["evidence"]["sector_quote"]
            st.markdown(f"**Sector cue** — page {sq['page']}")
            st.markdown(f"> {sq['quote']}")

        if ai["assumptions"]["adjustment_rules_fired"]:
            st.markdown("**Adjustment rules triggered (with quotes):**")
            for h in ai["assumptions"]["adjustment_rules_fired"]:
                st.markdown(f"- **{h['name']}** (page {h['page']}) — multiplier={h['jobs_per_million_mult']} | Δdirect_share={h['direct_share_delta']:+.02f}")
                st.markdown(f"  > {h['quote']}")
        else:
            st.markdown("_No special adjustment rules triggered from PAD text._")

        st.info(
            "AI fallback estimates are heuristic and for scoping only. "
            "Use PAD-explicit indicators or task-team–validated figures for formal reporting."
        )

        # Download
        out = {
            "mode": ai["mode"],
            "sector": ai["sector"],
            "investment_musd": ai["investment_musd"],
            "direct_jobs_est": ai["direct"],
            "indirect_jobs_est": ai["indirect"],
            "uncertainty_pct": ai["uncertainty_pct"],
            "confidence": ai["confidence"],
            "assumption_jobs_per_million_total": ai["assumptions"]["jobs_per_million_total"],
            "assumption_direct_share": ai["assumptions"]["direct_share"],
            "amount_evidence_page": ai["evidence"]["amount_quote"].get("page") if ai["evidence"].get("amount_quote") else None,
            "amount_evidence_quote": ai["evidence"]["amount_quote"].get("quote") if ai["evidence"].get("amount_quote") else None,
            "sector_evidence_page": ai["evidence"]["sector_quote"]["page"] if ai["evidence"].get("sector_quote") else None,
            "sector_evidence_quote": ai["evidence"]["sector_quote"]["quote"] if ai["evidence"].get("sector_quote") else None,
            "adjustments_json": str(ai["assumptions"]["adjustment_rules_fired"]),
        }
        df = pd.DataFrame([out])
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="AI-fallback")
        st.download_button(
            "Download as Excel",
            data=buf.getvalue(),
            file_name="pad_jobs_ai_fallback.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("Upload a PAD PDF to begin.")
