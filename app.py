
# app.py 
# PAD Job Creation & Better Jobs Analyzer — Robust Ensemble + Monte Carlo with traceable assumptions (v2 — condensed evidence + lower uncertainty)
import re
from io import BytesIO
from typing import Tuple, Dict, List, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
# ============================================================
# Streamlit Config & Header
# ============================================================
st.set_page_config(page_title="PAD Jobs & Better Jobs Analyzer", layout="wide", page_icon="📄")
st.title("PAD Job Creation & Better Jobs Analyzer")
st.caption(
    "This app (1) extracts **explicit** jobs (direct/indirect) where stated in PADs, and "
    "(2) when absent, **estimates** them using a robust ensemble model with Monte Carlo and evidence-driven shrinkage. "
    "In both cases, it estimates **Better Jobs** (what makes them 'better' is explained with PAD signals). "
    "All estimates remain traceable to text cues and page-level evidence."
)
# ============================================================
# PDF & Text Helpers
# ============================================================
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
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
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

# Short snippet helper
def short_snip(s: str, max_len: int = 160) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len-1].rstrip() + "…"

# ============================================================
# Investment Amount Parsing (US$ million only; robust to wording)
# ============================================================
INV_PATTERNS: List[re.Pattern] = [
    re.compile(r"(US\$\s*|USD\s*|\$)\s*([0-9][\d,]*\.?\d*)\s*(million|billion)", re.I),
    re.compile(r"([0-9][\d,]*\.?\d*)\s*(million|billion)\s*(US\$|USD|\$)?", re.I),
    re.compile(r"(total (?:project )?cost|financing(?: amount)?|loan amount)\D{0,40}(US\$|USD|\$)?\s*([0-9][\d,]*\.?\d*)\s*(million|billion)", re.I),
]

def parse_investment_amount_million(text: str) -> Tuple[Optional[float], str, Dict[str, Any]]:
    """
    Returns (amount_in_million, confidence_label, evidence_dict)
    evidence_dict = { 'quote': ..., 'page': ... } when available (page added by caller).
    """
    candidates: List[Tuple[float, str]] = []
    for pat in INV_PATTERNS:
        for m in pat.finditer(text):
            try:
                groups = m.groups()
                # Find numeric and unit
                nums = [g for g in groups if g and re.match(r"^[0-9][\d,]*\.?\d*$", g)]
                units = [g for g in groups if g and re.search(r"(million|billion)", g, re.I)]
                if not nums or not units:
                    continue
                num = nums[0]
                unit = units[0]
                val = float(num.replace(",", ""))
                if "billion" in unit.lower():
                    val *= 1000.0
                snippet = m.group(0)
                candidates.append((val, snippet))
            except Exception:
                continue
    if not candidates:
        # generic fallback
        m = re.search(r"([0-9][\d,]*\.?\d*)\s*(million|billion)", text, flags=re.I)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if m.group(2).lower() == "billion":
                    val *= 1000.0
                return val, "Medium (generic amount found)", {"quote": m.group(0), "page": None}
            except Exception:
                pass
        return None, "Low (no clear amount found)", {"quote": "No clear financing amount detected.", "page": None}
    # choose highest amount (often total financing)
    best_val, best_snip = max(candidates, key=lambda x: x[0])
    distinct_vals = {round(v, 2) for v, _ in candidates}
    conf = "High" if len(distinct_vals) == 1 else "Medium (multiple amounts detected)"
    return best_val, conf, {"quote": best_snip, "page": None}

def map_amount_to_page_evidence(pages: List[str], snippet: str) -> Optional[Dict[str, Any]]:
    if not snippet:
        return None
    for i, page in enumerate(pages):
        if snippet in page:
            return {"page": i + 1, "quote": snippet}
    return None

# ============================================================
# Explicit Job Extraction (keeps your robust logic, expanded)
# ============================================================
NON_JOB_UNITS = r"(person[\-\s]?days?|man[\-\s]?days?|worker[\-\s]?days?|job[\-\s]?years?|job[\-\s]?months?)"
NEAR_EXCLUSION = re.compile(NON_JOB_UNITS, re.I)
PAIR_PATTERNS = [
    re.compile(
        r"(?P<direct>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)\s+(?:and|&|,)\s+(?P<indirect>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)",
        re.I
    ),
    re.compile(
        r"(?P<indirect>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)\s+(?:and|&|,)\s+(?P<direct>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)",
        re.I
    ),
    re.compile(
        r"(?P<total>\d[\d,\.]*)\s+(?:jobs?|employment)\b[^\.\n]*?\bof which\b[^\.\n]*?(?P<direct>\d[\d,\.]*)\s+direct[^\.\n]*?(?:and|,)\s+(?P<indirect>\d[\d,\.]*)\s+indirect",
        re.I
    ),
]
DIRECT_PATTERNS = [
    re.compile(r"(?P<num>\d[\d,\.]*)\s+direct\s+(?:jobs?|employment|FTEs?)\b", re.I),
    re.compile(r"(?:jobs?|employment|FTEs?)\s+direct(?:ly)?\s*(?P<num>\d[\d,\.]*)\b", re.I),
]
INDIRECT_PATTERNS = [
    re.compile(r"(?P<num>\d[\d,\.]*)\s+indirect\s+(?:jobs?|employment|FTEs?)\b", re.I),
]
TOTAL_PATTERNS = [
    re.compile(r"(?P<num>\d[\d,\.]*)\s+(?:jobs?|employment)\s+(?:created|generated|supported|expected|targets?)\b", re.I),
    re.compile(r"total\s+(?:jobs?|employment)\s*(?:=|:)\s*(?P<num>\d[\d,\.]*)\b", re.I),
]
SECTION_RANKS = [
    (re.compile(r"results framework|annex\s+1[:\s\-]*\s*results", re.I), 0),
    (re.compile(r"pdo indicators?|project development objective", re.I), 1),
    (re.compile(r"economic analysis|annex\s+4[:\s\-]*\s*economic", re.I), 2),
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
            "evidence": {"jobs_quote": {"page": best["page"], "quote": best["quote"]}}
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
            candidates.sort(key=lambda t: (t[0], t[1], min(t[2]["page"], t[3]["page"])) )
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
    return None

# ============================================================
# Sector & Signals (Multi-label detection + quality signals)
# ============================================================
SECTOR_CUES: Dict[str, List[str]] = {
    "Agriculture": ["agriculture", "agri-", "agro", "irrigation", "farm", "rural livelihoods", "horticulture", "livestock"],
    "Manufacturing": ["manufactur", "factory", "industrial park", "industry 4.0", "value chain", "cluster"],
    "Energy": ["energy", "electric", "power", "generation", "grid", "renewable", "solar", "wind", "hydro", "geothermal"],
    "Transport": ["transport", "road", "highway", "rail", "port", "logistics", "corridor"],
    "ICT": ["ict", "digital", "broadband", "connectivity", "data center", "e-government", "platform"],
    "Health": ["health", "clinic", "hospital", "public health", "disease"],
    "Education": ["education", "school", "tvet", "teacher", "learning", "curriculum", "skills"],
    "Finance / Private Sector": ["msme", "sme", "finance", "credit", "lending", "bank", "guarantee", "matching grant", "credit line"],
    "Water": ["water", "sanitation", "wastewater", "wss", "utility"],
    "Urban": ["urban", "municipal", "city", "housing", "land use"],
    "Social Protection": ["social protection", "cash transfer", "safety net", "public works", "cash-for-work", "labor-intensive"],
    "Other / General": [],
}

# Base priors per sector (total jobs per million & direct share)
# These are used as priors and will be adjusted and shrunk by signals & evidence quality.
AI_PRIORS: Dict[str, Dict[str, float]] = {
    "Agriculture": {"jobs_per_million_total": 170.0, "direct_share": 0.36},
    "Manufacturing": {"jobs_per_million_total": 210.0, "direct_share": 0.42},
    "Energy": {"jobs_per_million_total": 65.0, "direct_share": 0.30},
    "Transport": {"jobs_per_million_total": 85.0, "direct_share": 0.34},
    "ICT": {"jobs_per_million_total": 75.0, "direct_share": 0.36},
    "Health": {"jobs_per_million_total": 105.0, "direct_share": 0.35},
    "Education": {"jobs_per_million_total": 95.0, "direct_share": 0.35},
    "Finance / Private Sector": {"jobs_per_million_total": 135.0, "direct_share": 0.27},
    "Water": {"jobs_per_million_total": 90.0, "direct_share": 0.31},
    "Urban": {"jobs_per_million_total": 115.0, "direct_share": 0.35},
    "Social Protection": {"jobs_per_million_total": 190.0, "direct_share": 0.47},
    "Other / General": {"jobs_per_million_total": 115.0, "direct_share": 0.35},
}

# Signals that influence both quantity (jobs per $) and "quality" (better-job probability)
ADJUSTMENT_RULES: List[Dict[str, Any]] = [
    {
        "name": "Labor-intensive / public works",
        "pattern": re.compile(r"\b(labor[\-\s]?intensive|public works|cash[\-\s]?for[\-\s]?work)\b", re.I),
        "jobs_per_million_mult": 1.4,
        "direct_share_delta": +0.05,
        "better_job_delta_direct": -0.04,
        "better_job_delta_indirect": -0.02,
        "quality_note": "Short-term, low-wage unless paired with standards/skills."
    },
    {
        "name": "MSME finance / credit line",
        "pattern": re.compile(r"\b(msme|sme|credit line|matching grant|partial credit guarantee)\b", re.I),
        "jobs_per_million_mult": 1.15,
        "direct_share_delta": -0.08,
        "better_job_delta_direct": +0.03,
        "better_job_delta_indirect": +0.02,
        "quality_note": "Access to finance can raise productivity/formality."
    },
    {
        "name": "Skills, training, apprenticeships",
        "pattern": re.compile(r"\b(skills|training|upskilling|reskilling|apprenticeship|certification|tvet)\b", re.I),
        "jobs_per_million_mult": 1.05,
        "direct_share_delta": +0.00,
        "better_job_delta_direct": +0.07,
        "better_job_delta_indirect": +0.03,
        "quality_note": "Skills raise wages, employability and productivity."
    },
    {
        "name": "Labor standards / formalization / social protection",
        "pattern": re.compile(r"\b(formaliz|labor standards?|occupational|social protection|social insurance|compliance)\b", re.I),
        "jobs_per_million_mult": 1.00,
        "direct_share_delta": +0.00,
        "better_job_delta_direct": +0.10,
        "better_job_delta_indirect": +0.05,
        "quality_note": "Contracts, benefits, OSH compliance improve job quality."
    },
    {
        "name": "OSH / safety investments",
        "pattern": re.compile(r"\b(occupational safety|OSH|workplace safety|safety equipment|PPE)\b", re.I),
        "jobs_per_million_mult": 1.00,
        "direct_share_delta": +0.00,
        "better_job_delta_direct": +0.06,
        "better_job_delta_indirect": +0.02,
        "quality_note": "Safer workplaces are a core dimension of better jobs."
    },
    {
        "name": "Female employment / childcare / inclusion",
        "pattern": re.compile(r"\b(gender|women|female|childcare|care services|inclusive|disability)\b", re.I),
        "jobs_per_million_mult": 1.02,
        "direct_share_delta": +0.00,
        "better_job_delta_direct": +0.05,
        "better_job_delta_indirect": +0.03,
        "quality_note": "Inclusion policies correlate with better conditions and retention."
    },
    {
        "name": "Green / renewable / resource efficiency",
        "pattern": re.compile(r"\b(solar|photovoltaic|wind|hydro|geothermal|energy efficiency|green|climate-smart)\b", re.I),
        "jobs_per_million_mult": 0.95,
        "direct_share_delta": -0.02,
        "better_job_delta_direct": +0.03,
        "better_job_delta_indirect": +0.02,
        "quality_note": "Green sectors often require higher skills & standards."
    },
    {
        "name": "Construction heavy / civil works",
        "pattern": re.compile(r"\b(civil works|construction|rehabilitation|km of road|bridges?)\b", re.I),
        "jobs_per_million_mult": 1.10,
        "direct_share_delta": +0.04,
        "better_job_delta_direct": +0.00,
        "better_job_delta_indirect": +0.00,
        "quality_note": "Construction raises direct jobs; quality neutral unless standards present."
    },
    {
        "name": "Technical assistance emphasis",
        "pattern": re.compile(r"\b(technical assistance|TA component|advisory)\b", re.I),
        "jobs_per_million_mult": 0.85,
        "direct_share_delta": -0.03,
        "better_job_delta_direct": +0.02,
        "better_job_delta_indirect": +0.02,
        "quality_note": "TA boosts capabilities; smaller job counts per $ but can improve quality."
    },
]

# Base better-jobs probability priors (by sector)
BETTER_JOBS_PRIOR = {
    "Agriculture": 0.32,
    "Manufacturing": 0.40,
    "Energy": 0.44,
    "Transport": 0.36,
    "ICT": 0.46,
    "Health": 0.42,
    "Education": 0.41,
    "Finance / Private Sector": 0.38,
    "Water": 0.39,
    "Urban": 0.37,
    "Social Protection": 0.30,
    "Other / General": 0.36
}

# --- NEW: sector evidence weighting by section rank + priority count ---
def detect_sectors_weighted(pages: List[str]) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]], int]:
    """
    Returns: (list of (sector, weight) for top 2 sectors summing to 1.0,
              evidence items [{sector, page, quote, rank}],
              priority_hits = count of sector cues found on priority pages (rank <= 1.5)).
    """
    def weight_from_rank(r: float) -> float:
        if r <= 0.5:
            return 2.0
        if r <= 1.5:
            return 1.6
        if r <= 2.5:
            return 1.25
        return 1.0

    scores: Dict[str, float] = {}
    evidence: List[Dict[str, Any]] = []
    priority_hits = 0
    for i, page in enumerate(pages):
        low = page.lower()
        r = page_rank(pages, i)
        w = weight_from_rank(r)
        for sector, cues in SECTOR_CUES.items():
            for cue in cues:
                if cue and cue in low:
                    cnt = low.count(cue)
                    scores[sector] = scores.get(sector, 0.0) + w * cnt
                    # capture one example per sector
                    if not any(ev.get("sector") == sector for ev in evidence):
                        m = re.search(re.escape(cue), low, re.I)
                        if m:
                            q = exact_sentence(page, m.span())
                            evidence.append({"sector": sector, "page": i + 1, "quote": q, "rank": r})
                            if r <= 1.5:
                                priority_hits += 1
    if not scores:
        return [("Other / General", 1.0)], evidence, priority_hits
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:2]
    total = sum(s for _, s in top)
    weighted = [(sec, s / total if total > 0 else 0.5) for sec, s in top]
    return weighted, evidence, priority_hits

# include rank + impact in adjustments evidence

def detect_adjustments(pages: List[str]) -> List[Dict[str, Any]]:
    hits = []
    for i, page in enumerate(pages):
        r = page_rank(pages, i)
        for rule in ADJUSTMENT_RULES:
            for m in rule["pattern"].finditer(page):
                sent = exact_sentence(page, m.span())
                impact = abs(np.log(max(1e-6, rule["jobs_per_million_mult"]))) \
                         + 2.0 * abs(rule["direct_share_delta"]) \
                         + 1.2 * max(0.0, rule["better_job_delta_direct"]) \
                         + 0.8 * max(0.0, rule["better_job_delta_indirect"])
                hits.append({
                    "name": rule["name"],
                    "jobs_per_million_mult": rule["jobs_per_million_mult"],
                    "direct_share_delta": rule["direct_share_delta"],
                    "better_job_delta_direct": rule["better_job_delta_direct"],
                    "better_job_delta_indirect": rule["better_job_delta_indirect"],
                    "quality_note": rule["quality_note"],
                    "page": i + 1,
                    "quote": sent,
                    "rank": r,
                    "impact": float(impact),
                })
    # deduplicate (name, quote, page)
    seen = set()
    uniq = []
    for h in hits:
        key = (h["name"], h["quote"], h["page"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq

# ============================================================
# Robust Estimation (Ensemble + Monte Carlo)
# ============================================================

def compute_uncertainty_scale(amount_conf: str, sector_evidence_count: int, n_rules: int, priority_hits: int = 0) -> Dict[str, float]:
    """
    Returns dispersion controls for Monte Carlo. Lower sigma => tighter (reduced) uncertainty.
    """
    # base dispersions (slightly tighter than v1)
    sigma_jobs = 0.38  # lognormal sigma for jobs_per_million (base)
    k_direct = (18, 24)  # Beta parameters base for direct share (slightly stronger than v1)

    # amount confidence effect
    if amount_conf.startswith("High"):
        sigma_jobs -= 0.12
    elif amount_conf.startswith("Medium"):
        sigma_jobs -= 0.05
    else:
        sigma_jobs += 0.05

    # sector evidence effect
    if sector_evidence_count >= 2:
        sigma_jobs -= 0.06
    elif sector_evidence_count == 1:
        sigma_jobs -= 0.03
    else:
        sigma_jobs += 0.05

    # rule corroboration effect
    if n_rules >= 3:
        sigma_jobs -= 0.06
    elif n_rules >= 1:
        sigma_jobs -= 0.02
    else:
        sigma_jobs += 0.04

    # NEW: priority cues (Results/PDO/Economic Analysis pages) -> more shrinkage
    if priority_hits >= 2:
        sigma_jobs -= 0.10
    elif priority_hits == 1:
        sigma_jobs -= 0.05

    # clamp
    sigma_jobs = float(np.clip(sigma_jobs, 0.10, 0.50))

    # Direct share concentration via Beta prior strength
    base_alpha, base_beta = k_direct
    strength_delta = 10 if n_rules >= 3 else (6 if n_rules >= 1 else 0)
    if amount_conf.startswith("High"):
        strength_delta += 6
    if sector_evidence_count >= 2:
        strength_delta += 6
    if priority_hits >= 1:
        strength_delta += 6

    alpha = base_alpha + strength_delta
    beta = base_beta + strength_delta
    return {"sigma_jobs": sigma_jobs, "alpha_direct": alpha, "beta_direct": beta}


def ai_estimate_jobs_and_better(pages: List[str]) -> Dict[str, Any]:
    full_text = "\n".join(pages)
    # 1) Sector detection (weighted + priority)
    sector_weights, sector_evs, priority_hits = detect_sectors_weighted(pages)
    sector_ev_present = len(sector_evs) > 0

    # 2) Financing amount parsing
    amount_m, amount_conf, amount_ev = parse_investment_amount_million(full_text)
    if amount_ev and not amount_ev.get("page"):
        mapped = map_amount_to_page_evidence(pages, amount_ev.get("quote", ""))
        if mapped:
            amount_ev["page"] = mapped["page"]
            amount_ev["quote"] = mapped["quote"]
    used_default_amount = False
    if amount_m is None:
        amount_m = 50.0  # default
        used_default_amount = True
        amount_conf = "Low (used default US$50M because financing not detected)"
        amount_ev = {"page": None, "quote": "No clear financing amount found; default used for AI estimate."}

    # 3) Initialize priors as weighted average across sectors
    def wavg(key: str, table: Dict[str, Dict[str, float]]) -> float:
        out = 0.0
        for sec, w in sector_weights:
            out += w * table.get(sec, table["Other / General"]).get(key, table["Other / General"][key])
        return out

    base_jobs_per_million = wavg("jobs_per_million_total", AI_PRIORS)
    base_direct_share = wavg("direct_share", AI_PRIORS)

    base_better_prior = 0.0
    for sec, w in sector_weights:
        base_better_prior += w * BETTER_JOBS_PRIOR.get(sec, BETTER_JOBS_PRIOR["Other / General"])

    # 4) Adjustment rules from PAD text (with dampening to avoid compounding and reduce uncertainty)
    adj_hits = detect_adjustments(pages)
    # sort by section rank (more authoritative first)
    adj_sorted = sorted(adj_hits, key=lambda h: (h.get("rank", DEFAULT_RANK), h.get("page", 9999)))
    # dampening weights for sequential application
    damp_seq = [1.0, 0.75, 0.6, 0.5, 0.4, 0.35, 0.3]

    jobs_per_million = base_jobs_per_million
    direct_share = base_direct_share
    better_p_direct = base_better_prior
    better_p_indirect = base_better_prior - 0.03  # often slightly lower upstream in supply chains

    # combine multipliers with dampening
    combined_mult = 1.0
    for k, h in enumerate(adj_sorted):
        w = damp_seq[k] if k < len(damp_seq) else 0.3
        combined_mult *= (1.0 + w * (h["jobs_per_million_mult"] - 1.0))
        direct_share = float(np.clip(direct_share + w * h["direct_share_delta"], 0.05, 0.90))
        better_p_direct = float(np.clip(better_p_direct + w * h["better_job_delta_direct"], 0.05, 0.95))
        better_p_indirect = float(np.clip(better_p_indirect + w * h["better_job_delta_indirect"], 0.05, 0.95))
    # soft shrinkage of combined multiplier toward 1.0 when evidence is weak
    evidence_strength = (min(3, len(adj_sorted)) * 0.25) + (min(2, priority_hits) * 0.2) + (0.2 if amount_conf.startswith("High") else (0.1 if amount_conf.startswith("Medium") else 0.0)) + (0.15 if len(sector_evs) >= 2 else (0.05 if len(sector_evs) == 1 else 0.0))
    evidence_strength = float(np.clip(evidence_strength, 0.35, 0.95))
    jobs_per_million = base_jobs_per_million * (1 - (1 - combined_mult) * evidence_strength)

    # guardrails on jobs_per_million (winsorize point estimate against priors)
    lower_guard = 0.5 * base_jobs_per_million
    upper_guard = 3.0 * base_jobs_per_million
    jobs_per_million = float(np.clip(jobs_per_million, lower_guard, upper_guard))

    # 5) Evidence-driven uncertainty controls
    scales = compute_uncertainty_scale(amount_conf, len(sector_evs), len(adj_sorted), priority_hits)
    sigma_jobs = scales["sigma_jobs"]
    alpha_direct = scales["alpha_direct"]
    beta_direct = scales["beta_direct"]

    # 6) Monte Carlo simulation (reduced uncertainty with stronger evidence)
    N = 8000
    # Lognormal for jobs_per_million (mean at current point)
    mu = np.log(max(jobs_per_million, 1e-6)) - 0.5 * sigma_jobs**2
    draws_jpm = np.random.lognormal(mean=mu, sigma=sigma_jobs, size=N)
    # winsorize tails to reduce undue influence of extreme values
    q01, q99 = np.quantile(draws_jpm, [0.01, 0.99])
    draws_jpm = np.clip(draws_jpm, q01, q99)

    # Beta for direct share (centered at current point)
    total_strength = alpha_direct + beta_direct
    mean_target = float(np.clip(direct_share, 0.05, 0.95))
    alpha_adj = mean_target * total_strength
    beta_adj = (1 - mean_target) * total_strength
    draws_direct_share = np.random.beta(alpha_adj, beta_adj, size=N)

    # Slight uncertainty around better job probabilities (narrower with more evidence + priority)
    bj_sigma_base = 0.07
    bj_sigma = bj_sigma_base - min(0.05, 0.012 * (len(adj_sorted) + len(sector_evs) + priority_hits))
    bj_sigma = float(np.clip(bj_sigma, 0.015, bj_sigma_base))
    draws_better_direct = np.clip(np.random.normal(loc=better_p_direct, scale=bj_sigma, size=N), 0.02, 0.98)
    draws_better_indirect = np.clip(np.random.normal(loc=better_p_indirect, scale=bj_sigma, size=N), 0.02, 0.98)

    # 7) Compute distributions
    total_jobs_draws = amount_m * draws_jpm
    direct_jobs_draws = np.round(total_jobs_draws * draws_direct_share)
    indirect_jobs_draws = np.round(total_jobs_draws - direct_jobs_draws)
    better_direct_draws = np.round(direct_jobs_draws * draws_better_direct)
    better_indirect_draws = np.round(indirect_jobs_draws * draws_better_indirect)

    def pct(arr, q):
        return float(np.percentile(arr, q))

    out = {
        "mode": "AI-fallback",
        "sector_weights": sector_weights,
        "investment_musd": round(float(amount_m), 2),
        "assumptions": {
            "base_jobs_per_million": round(base_jobs_per_million, 2),
            "base_direct_share": round(base_direct_share, 3),
            "base_better_prior": round(base_better_prior, 3),
            "adj_jobs_per_million_point": round(jobs_per_million, 2),
            "adj_direct_share_point": round(direct_share, 3),
            "adj_better_p_direct_point": round(better_p_direct, 3),
            "adj_better_p_indirect_point": round(better_p_indirect, 3),
            "uncertainty_controls": {
                "sigma_jobs": sigma_jobs,
                "alpha_direct": alpha_direct,
                "beta_direct": beta_direct,
                "bj_sigma": bj_sigma,
            },
            "adjustment_rules_fired": [{
                "name": h["name"],
                "jobs_per_million_mult": h["jobs_per_million_mult"],
                "direct_share_delta": h["direct_share_delta"],
                "better_job_delta_direct": h["better_job_delta_direct"],
                "better_job_delta_indirect": h["better_job_delta_indirect"],
                "quality_note": h["quality_note"],
                "page": h["page"],
                "quote": h["quote"],
                "rank": h.get("rank", DEFAULT_RANK),
                "impact": h.get("impact", 0.0),
            } for h in adj_sorted]
        },
        "distributions": {
            "total_jobs": {"p10": pct(total_jobs_draws, 10), "p50": pct(total_jobs_draws, 50), "p90": pct(total_jobs_draws, 90)},
            "direct_jobs": {"p10": pct(direct_jobs_draws, 10), "p50": pct(direct_jobs_draws, 50), "p90": pct(direct_jobs_draws, 90)},
            "indirect_jobs": {"p10": pct(indirect_jobs_draws, 10), "p50": pct(indirect_jobs_draws, 50), "p90": pct(indirect_jobs_draws, 90)},
            "better_direct": {"p10": pct(better_direct_draws, 10), "p50": pct(better_direct_draws, 50), "p90": pct(better_direct_draws, 90)},
            "better_indirect": {"p10": pct(better_indirect_draws, 10), "p50": pct(better_indirect_draws, 50), "p90": pct(better_indirect_draws, 90)},
        },
        "confidence": "Medium" if not used_default_amount else "Low",
        "evidence": {
            "sector_quotes": sector_evs if sector_ev_present else None,
            "amount_quote": amount_ev,
            "note": "Estimates derived from sector weights, financing, and PAD signals; see explanations below.",
            "priority_hits": priority_hits,
        }
    }

    # Compact ±% uncertainty around medians for display
    med = out["distributions"]["total_jobs"]["p50"]
    halfspan = (out["distributions"]["total_jobs"]["p90"] - out["distributions"]["total_jobs"]["p10"]) / 2.0
    plus_minus_pct = int(np.clip(100.0 * halfspan / med if med > 0 else 50.0, 6, 40))
    out["uncertainty_pct"] = plus_minus_pct
    return out

# ============================================================
# Better Jobs estimate for PAD-explicit counts
# ============================================================

def estimate_better_from_explicit(direct: Optional[int], indirect: Optional[int], pages: List[str]) -> Dict[str, Any]:
    # Reuse sector weights & signals to compute better-job probabilities
    sector_weights, sector_evs, priority_hits = detect_sectors_weighted(pages)
    adj_hits = detect_adjustments(pages)
    base_better_prior = 0.0
    for sec, w in sector_weights:
        base_better_prior += w * BETTER_JOBS_PRIOR.get(sec, BETTER_JOBS_PRIOR["Other / General"])
    better_p_direct = base_better_prior
    better_p_indirect = base_better_prior - 0.03
    # dampening as in AI fallback
    adj_sorted = sorted(adj_hits, key=lambda h: (h.get("rank", DEFAULT_RANK), h.get("page", 9999)))
    damp_seq = [1.0, 0.75, 0.6, 0.5, 0.4, 0.35, 0.3]
    for k, h in enumerate(adj_sorted):
        w = damp_seq[k] if k < len(damp_seq) else 0.3
        better_p_direct = float(np.clip(better_p_direct + w * h["better_job_delta_direct"], 0.02, 0.98))
        better_p_indirect = float(np.clip(better_p_indirect + w * h["better_job_delta_indirect"], 0.02, 0.98))
    # Slight uncertainty around better-job probabilities (narrow if many cues + priority)
    bj_sigma = 0.06 - min(0.04, 0.01 * (len(adj_sorted) + len(sector_evs) + priority_hits))
    bj_sigma = float(np.clip(bj_sigma, 0.015, 0.06))
    N = 4000
    draws_better_direct = np.clip(np.random.normal(better_p_direct, bj_sigma, size=N), 0.02, 0.98)
    draws_better_indirect = np.clip(np.random.normal(better_p_indirect, bj_sigma, size=N), 0.02, 0.98)

    def pct(arr, q): return float(np.percentile(arr, q))
    out = {"evidence": {"sector_quotes": sector_evs or None, "adjustment_quotes": adj_sorted or None}}
    if direct is not None:
        bd = np.round(direct * draws_better_direct)
        out["better_direct"] = {"p10": pct(bd, 10), "p50": pct(bd, 50), "p90": pct(bd, 90), "p": better_p_direct}
    if indirect is not None:
        bi = np.round(indirect * draws_better_indirect)
        out["better_indirect"] = {"p10": pct(bi, 10), "p50": pct(bi, 50), "p90": pct(bi, 90), "p": better_p_indirect}
    return out

# ============================================================
# NEW: Explanations & Better-Job forms helpers (with concise snippets)
# ============================================================

RULE_TO_FORM = {
    "Skills, training, apprenticeships": "Better wages & productivity",
    "MSME finance / credit line": "Better wages & productivity",
    "Technical assistance emphasis": "Better wages & productivity",
    "Labor standards / formalization / social protection": "Better terms of employment (contracts, benefits)",
    "OSH / safety investments": "Safer working conditions (OSH)",
    "Female employment / childcare / inclusion": "Inclusion & accessibility",
    "Green / renewable / resource efficiency": "Greener production & higher standards",
}

DEFAULT_FORM_WEIGHTS = {
    "Better wages & productivity": 0.55,
    "Better terms of employment (contracts, benefits)": 0.25,
    "Safer working conditions (OSH)": 0.12,
    "Inclusion & accessibility": 0.06,
    "Greener production & higher standards": 0.02,
}

def _fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"

def _fmt_pp(x: float) -> str:
    return f"{x*100:+.0f} pp"

# concise evidence selectors

def top_sector_examples(sector_evs: List[Dict[str, Any]], sector_weights: List[Tuple[str, float]], top_n: int = 2) -> List[Dict[str, Any]]:
    top_sectors = [s for s, _ in sorted(sector_weights, key=lambda t: t[1], reverse=True)[:top_n]]
    out = []
    for s in top_sectors:
        items = [ev for ev in sector_evs if ev.get("sector") == s]
        if items:
            out.append(items[0])
    return out

def top_adjustment_hits(adj: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    if not adj:
        return []
    ranked = sorted(adj, key=lambda h: (-(h.get("impact", 0.0)), h.get("rank", DEFAULT_RANK), h.get("page", 9e9)))
    return ranked[:top_n]


def build_adjustment_explanations(sector_weights: List[Tuple[str, float]],
                                  base_jobs_per_million: float,
                                  base_direct_share: float,
                                  base_better_direct: float,
                                  base_better_indirect: float,
                                  adj_hits: List[Dict[str, Any]]) -> List[str]:
    """Legacy detailed explanation kept (not shown by default in v2)."""
    explanations = []
    jpm = base_jobs_per_million
    dshare = base_direct_share
    bj_d = base_better_direct
    bj_i = base_better_indirect
    for h in adj_hits:
        new_jpm = jpm * h["jobs_per_million_mult"]
        new_dshare = float(np.clip(dshare + h["direct_share_delta"], 0.05, 0.90))
        new_bj_d = float(np.clip(bj_d + h["better_job_delta_direct"], 0.05, 0.95))
        new_bj_i = float(np.clip(bj_i + h["better_job_delta_indirect"], 0.05, 0.95))
        msg = (
            f"**{h['name']}** (page {h['page']}): increased jobs-per-$ from {jpm:.1f} → {new_jpm:.1f} "
            f"({ _fmt_pct(h['jobs_per_million_mult']-1) if h['jobs_per_million_mult']!=1 else '+0%'}), "
            f"shifted direct share {dshare:.2f} → {new_dshare:.2f} ({ _fmt_pp(new_dshare-dshare)}), "
            f"and adjusted 'better job' probabilities to direct {bj_d:.2f} → {new_bj_d:.2f} ({ _fmt_pp(new_bj_d-bj_d)}), "
            f"indirect {bj_i:.2f} → {new_bj_i:.2f} ({ _fmt_pp(new_bj_i-bj_i)}). _{h['quality_note']}_."
        )
        explanations.append("".join(msg))
        jpm, dshare, bj_d, bj_i = new_jpm, new_dshare, new_bj_d, new_bj_i
    if not explanations:
        explanations.append("No special adjustment signals detected; estimates rely on sector priors and financing amount.")
    return explanations


def explain_sector_basis(sector_weights: List[Tuple[str, float]]) -> Tuple[str, float, float]:
    parts = []
    base_jpm = 0.0
    base_dir = 0.0
    for sec, w in sector_weights:
        pri = AI_PRIORS.get(sec, AI_PRIORS["Other / General"])
        parts.append(f"{sec} (weight {w:.0%} \n priors: {pri['jobs_per_million_total']:.0f} jobs/US$1M, direct share {pri['direct_share']:.2f})")
        base_jpm += w * pri["jobs_per_million_total"]
        base_dir += w * pri["direct_share"]
    text = (
        "**Sector composition**: The model detected sector cues and weighted the base rates accordingly → "
        f"base jobs-per-$ ≈ {base_jpm:.1f} and base direct share ≈ {base_dir:.2f}. "
        "Weights come from frequency of sector cues in the PAD."
    )
    text += "\n\n• " + "; ".join(parts)
    return text, base_jpm, base_dir


def build_better_forms(better_direct_p50: Optional[float], better_indirect_p50: Optional[float],
                       adj_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total_better = (better_direct_p50 or 0) + (better_indirect_p50 or 0)
    scores: Dict[str, float] = {k: 0.0 for k in DEFAULT_FORM_WEIGHTS.keys()}
    provenance: Dict[str, List[int]] = {k: [] for k in DEFAULT_FORM_WEIGHTS.keys()}
    for h in adj_hits:
        form = RULE_TO_FORM.get(h["name"])  # None for construction-heavy, etc.
        if not form:
            continue
        delta = max(0.0, h["better_job_delta_direct"]) + max(0.0, h["better_job_delta_indirect"])
        if delta <= 0:
            continue
        scores[form] += float(delta)
        provenance[form].append(h["page"])
    if sum(scores.values()) <= 1e-9:
        scores = DEFAULT_FORM_WEIGHTS.copy()
    ssum = sum(scores.values())
    shares = {k: (v/ssum if ssum > 0 else 0) for k, v in scores.items()}
    results = []
    for label, sh in shares.items():
        count = int(round(total_better * sh))
        pgs = provenance.get(label) or []
        rationale = (
            f"Driven by signals on pages {sorted(set(pgs))} "
            if pgs else "Driven by sector-average quality improvements (no specific signals detected)."
        )
        results.append({"form": label, "share": sh, "count": count, "rationale": rationale})
    results.sort(key=lambda x: (-x["count"], x["form"]))
    return results

# ============================================================
# UI Flow
# ============================================================
uploaded_file = st.file_uploader("Upload a PAD PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text..."):
        pages = extract_text_per_page(uploaded_file)
    if not any(pages):
        st.error("No selectable text extracted. The PDF may be scanned or image-only. Consider OCR.")
        st.stop()

    explicit = find_explicit_jobs(pages)

    if explicit:
        # ---------------------------- EXPLICIT MODE ---------------------------
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

        # Better Jobs based on signals even when jobs are explicit
        bj = estimate_better_from_explicit(explicit.get("direct"), explicit.get("indirect"), pages)
        st.subheader("Better Jobs (estimated from PAD signals)")
        c5, c6 = st.columns(2)
        if "better_direct" in bj:
            bd = bj["better_direct"]["p50"]
            rng = (bj["better_direct"]["p10"], bj["better_direct"]["p90"])
            with c5:
                st.metric("Better-Direct (P50)", f"{int(round(bd)):,}")
            st.caption(f"P10–P90: {int(round(rng[0])):,} – {int(round(rng[1])):,}")
        if "better_indirect" in bj:
            bi = bj["better_indirect"]["p50"]
            rngi = (bj["better_indirect"]["p10"], bj["better_indirect"]["p90"])
            with c6:
                st.metric("Better-Indirect (P50)", f"{int(round(bi)):,}")
            st.caption(f"P10–P90: {int(round(rngi[0])):,} – {int(round(rngi[1])):,}")

        # NEW: Forms in which jobs became "better"
        st.markdown("**How are these jobs 'better'? (forms)**")
        bd_p50 = bj.get("better_direct", {}).get("p50")
        bi_p50 = bj.get("better_indirect", {}).get("p50")
        forms = build_better_forms(bd_p50, bi_p50, bj.get("evidence", {}).get("adjustment_quotes") or [])
        df_forms = pd.DataFrame([{**f, "share": f"{f['share']*100:.0f}%"} for f in forms])[ ["form","count","share","rationale"] ]
        st.dataframe(df_forms, hide_index=True, use_container_width=True)

        # PAD Quotes
        st.subheader("Exact PAD Quotes")
        ev = explicit.get("evidence", {})
        if "jobs_quote" in ev and ev["jobs_quote"]:
            st.markdown(f"**Jobs statement** — page {ev['jobs_quote']['page']}")
            st.markdown(f"> {ev['jobs_quote']['quote']}")
        if bj["evidence"].get("sector_quotes"):
            st.markdown("**Sector cues (examples):**")
            for s in bj["evidence"]["sector_quotes"]:
                st.markdown(f"- **{s['sector']}** — page {s['page']}")
                st.markdown(f"  > {s['quote']}")
        if bj["evidence"].get("adjustment_quotes"):
            st.markdown("**Signals influencing Better Jobs:**")
            for h in bj["evidence"]["adjustment_quotes"]:
                st.markdown(
                    f"- **{h['name']}** (page {h['page']}) — {h['quality_note']}"
                )
                st.markdown(f"  > {h['quote']}")
        with st.expander("What counts as a 'Better Job' here?"):
            st.markdown(
                "- **Higher wages/productivity potential** (skills, certification, technology upgrading)\n"
                "- **Formality & social protection** (contracts, benefits, compliance)\n"
                "- **Workplace safety (OSH)** (safety systems, PPE, inspections)\n"
                "- **Inclusion** (women, disability, childcare support)\n"
                "- **Environmental quality** (green/clean jobs)\n\n"
                "The estimator reads PAD signals about these aspects and assigns probabilities to jobs being 'better'."
            )

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
            "better_forms_json": pd.Series(forms).to_json(orient="values")
        }
        if "better_direct" in bj:
            out.update({
                "better_direct_p10": int(round(bj["better_direct"]["p10"])),
                "better_direct_p50": int(round(bj["better_direct"]["p50"])),
                "better_direct_p90": int(round(bj["better_direct"]["p90"])),
                "better_direct_p": round(bj["better_direct"]["p"], 3)
            })
        if "better_indirect" in bj:
            out.update({
                "better_indirect_p10": int(round(bj["better_indirect"]["p10"])),
                "better_indirect_p50": int(round(bj["better_indirect"]["p50"])),
                "better_indirect_p90": int(round(bj["better_indirect"]["p90"])),
                "better_indirect_p": round(bj["better_indirect"]["p"], 3)
            })
        df = pd.DataFrame([out])
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="PAD-explicit+Better")
        st.download_button(
            "Download as Excel",
            data=buf.getvalue(),
            file_name="pad_jobs_explicit_better.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        # ---------------------------- AI FALLBACK MODE ------------------------
        ai = ai_estimate_jobs_and_better(pages)
        st.subheader("Job Creation (AI fallback, robust ensemble)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Direct Jobs (P50)", f"{int(round(ai['distributions']['direct_jobs']['p50'])):,}")
        st.caption(f"P10–P90: {int(round(ai['distributions']['direct_jobs']['p10'])):,} – {int(round(ai['distributions']['direct_jobs']['p90'])):,}")
        with c2:
            st.metric("Indirect Jobs (P50)", f"{int(round(ai['distributions']['indirect_jobs']['p50'])):,}")
        st.caption(f"P10–P90: {int(round(ai['distributions']['indirect_jobs']['p10'])):,} – {int(round(ai['distributions']['indirect_jobs']['p90'])):,}")
        with c3:
            st.metric("Total Jobs (P50)", f"{int(round(ai['distributions']['total_jobs']['p50'])):,}")
        st.caption(f"P10–P90: {int(round(ai['distributions']['total_jobs']['p10'])):,} – {int(round(ai['distributions']['total_jobs']['p90'])):,}")
        with c4:
            st.metric("Uncertainty (± around P50)", f"{ai['uncertainty_pct']}%")
        st.caption(
            f"Mode: {ai['mode']} • Investment used: US${ai['investment_musd']:.2f}M • "
            f"Sectors: {', '.join([f'{s} ({w:.0%})' for s, w in ai['sector_weights']])}"
        )

        # Better Jobs
        st.subheader("Better Jobs (estimated)")
        c5, c6 = st.columns(2)
        bd = ai["distributions"]["better_direct"]
        bi = ai["distributions"]["better_indirect"]
        with c5:
            st.metric("Better-Direct (P50)", f"{int(round(bd['p50'])):,}")
        st.caption(f"P10–P90: {int(round(bd['p10'])):,} – {int(round(bd['p90'])):,}")
        with c6:
            st.metric("Better-Indirect (P50)", f"{int(round(bi['p50'])):,}")
        st.caption(f"P10–P90: {int(round(bi['p10'])):,} – {int(round(bi['p90'])):,}")

        # NEW: Forms in which jobs became "better"
        st.markdown("**How are these jobs 'better'? (forms)**")
        forms = build_better_forms(bd.get('p50'), bi.get('p50'), ai["assumptions"]["adjustment_rules_fired"])
        df_forms = pd.DataFrame([{**f, "share": f"{f['share']*100:.0f}%"} for f in forms])[ ["form","count","share","rationale"] ]
        st.dataframe(df_forms, hide_index=True, use_container_width=True)

        # --- v2: Concise Evidence with Short Snippets ---
        st.subheader("PAD Evidence and Signals Used — concise (w/ short snippets)")
        # Financing amount (short)
        aq = ai["evidence"].get("amount_quote")
        if aq:
            page_note = f" (page {aq.get('page')})" if aq.get("page") else ""
            st.markdown("**Financing amount**" + page_note)
            st.markdown(f"- Used US${ai['investment_musd']:.2f}M to scale jobs; each extra US$1M adds ≈ {int(round(ai['assumptions']['adj_jobs_per_million_point']))} jobs.")
            if aq.get("quote"):
                st.markdown(f"  > {short_snip(aq['quote'])}")
            if "default" in ai.get("confidence", "").lower() or (aq.get("quote") and "default" in aq.get("quote").lower()):
                st.markdown("  - *No clear amount found; default US$50M used for scoping.*")
        
        # Sector basis (top examples only)
        if ai["evidence"].get("sector_quotes"):
            examples = top_sector_examples(ai["evidence"]["sector_quotes"], ai["sector_weights"], top_n=2)
            if examples:
                st.markdown("**Sector cues (top examples):**")
                for ex in examples:
                    st.markdown(f"- **{ex['sector']}** — page {ex['page']}")
                    st.markdown(f"  > {short_snip(ex['quote'])}")
        
        # Top signals by impact (condensed)
        adj = ai["assumptions"]["adjustment_rules_fired"]
        if adj:
            st.markdown("**Key signals shaping the estimate (top 3):**")
            for h in top_adjustment_hits(adj, top_n=3):
                dpp = _fmt_pp(h["direct_share_delta"]) if h["direct_share_delta"] else "+0 pp"
                jmult = h["jobs_per_million_mult"]
                jpct = _fmt_pct(jmult-1) if jmult != 1 else "+0%"
                bjd = _fmt_pp(h["better_job_delta_direct"]) if h["better_job_delta_direct"] else "+0 pp"
                bji = _fmt_pp(h["better_job_delta_indirect"]) if h["better_job_delta_indirect"] else "+0 pp"
                st.markdown(
                    f"- **{h['name']}** (p.{h['page']}): jobs/$ {jpct}, direct share {dpp}, better-job +{bjd} (direct) / +{bji} (indirect)."
                )
                st.markdown(f"  > {short_snip(h['quote'])}")
            extra = max(0, len(adj) - 3)
            if extra:
                st.caption(f"…and {extra} additional minor signals (omitted for brevity).")

        with st.expander("Details & methodology"):
            sect_text, base_jpm, base_dir = explain_sector_basis(ai["sector_weights"])
            st.markdown(sect_text)
            st.markdown("**Signal-by-signal effects (full):**")
            base_bj_d = ai["assumptions"]["base_better_prior"]
            base_bj_i = base_bj_d - 0.03
            for line in build_adjustment_explanations(ai["sector_weights"], base_jpm, base_dir, base_bj_d, base_bj_i, adj):
                st.markdown(f"- {line}")
            st.info("Monte Carlo simulation yields **P50** and **P10–P90** ranges. Uncertainty narrows as evidence strengthens.")

        st.info(
            "AI fallback estimates are for scoping and learning purposes. "
            "Use PAD-explicit indicators or task-team–validated figures for formal reporting."
        )

        # Download
        row = {
            "mode": ai["mode"],
            "investment_musd": ai["investment_musd"],
            "sector_weights": "; ".join([f"{s}:{w:.2f}" for s, w in ai["sector_weights"]]),
            "direct_p10": int(round(ai["distributions"]["direct_jobs"]["p10"])),
            "direct_p50": int(round(ai["distributions"]["direct_jobs"]["p50"])),
            "direct_p90": int(round(ai["distributions"]["direct_jobs"]["p90"])),
            "indirect_p10": int(round(ai["distributions"]["indirect_jobs"]["p10"])),
            "indirect_p50": int(round(ai["distributions"]["indirect_jobs"]["p50"])),
            "indirect_p90": int(round(ai["distributions"]["indirect_jobs"]["p90"])),
            "total_p10": int(round(ai["distributions"]["total_jobs"]["p10"])),
            "total_p50": int(round(ai["distributions"]["total_jobs"]["p50"])),
            "total_p90": int(round(ai["distributions"]["total_jobs"]["p90"])),
            "better_direct_p10": int(round(ai["distributions"]["better_direct"]["p10"])),
            "better_direct_p50": int(round(ai["distributions"]["better_direct"]["p50"])),
            "better_direct_p90": int(round(ai["distributions"]["better_direct"]["p90"])),
            "better_indirect_p10": int(round(ai["distributions"]["better_indirect"]["p10"])),
            "better_indirect_p50": int(round(ai["distributions"]["better_indirect"]["p50"])),
            "better_indirect_p90": int(round(ai["distributions"]["better_indirect"]["p90"])),
            "uncertainty_pct": ai["uncertainty_pct"],
            "confidence": ai["confidence"],
            "amount_evidence_page": ai["evidence"]["amount_quote"].get("page") if ai["evidence"].get("amount_quote") else None,
            "amount_evidence_quote": ai["evidence"]["amount_quote"].get("quote") if ai["evidence"].get("amount_quote") else None,
            "sector_evidence_json": str(ai["evidence"].get("sector_quotes") or []),
            "adjustments_json": str(ai["assumptions"]["adjustment_rules_fired"] or []),
            "better_forms_json": pd.Series(forms).to_json(orient="values"),
        }
        df = pd.DataFrame([row])
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="AI-fallback+Better")
        st.download_button(
            "Download as Excel",
            data=buf.getvalue(),
            file_name="pad_jobs_ai_better.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Upload a PAD PDF to begin.")
