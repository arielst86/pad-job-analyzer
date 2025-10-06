# app.py
# PAD Job Creation & Better Jobs Analyzer — Robust Ensemble + Monte Carlo with traceable assumptions

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
    "In both cases, it estimates **Better Jobs** (what makes them 'better' is explained with PAD quotes). "
    "All estimates remain traceable to text cues and page-level evidence."
)

# ============================================================
# PDF & Text Helpers
# ============================================================
def clean_text_basic(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"-\s*\n\s*", "", txt)   # de-hyphenate line breaks
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
            groups = m.groups()
            # normalize groups to find numeric and unit
            nums = [g for g in groups if g and re.match(r"^[0-9][\d,]*\.?\d*$", g)]
            units = [g for g in groups if g and re.search(r"(million|billion)", g, re.I)]
            if not nums or not units:
                continue
            num = nums[0]
            unit = units[0]
            try:
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
    re.compile(r"total\s+(?:jobs?|employment)\s*(?:=|:)?\s*(?P<num>\d[\d,\.]*)\b", re.I),
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
    "Energy": {"jobs_per_million_total": 65.0,  "direct_share": 0.30},
    "Transport": {"jobs_per_million_total": 85.0,  "direct_share": 0.34},
    "ICT": {"jobs_per_million_total": 75.0,  "direct_share": 0.36},
    "Health": {"jobs_per_million_total": 105.0, "direct_share": 0.35},
    "Education": {"jobs_per_million_total": 95.0,  "direct_share": 0.35},
    "Finance / Private Sector": {"jobs_per_million_total": 135.0, "direct_share": 0.27},
    "Water": {"jobs_per_million_total": 90.0,  "direct_share": 0.31},
    "Urban": {"jobs_per_million_total": 115.0, "direct_share": 0.35},
    "Social Protection": {"jobs_per_million_total": 190.0, "direct_share": 0.47},
    "Other / General": {"jobs_per_million_total": 115.0, "direct_share": 0.35},
}

# Signals that influence both quantity (jobs per $) and "quality" (better jobs probability)
# Each rule: regex -> multipliers and better-job deltas; with transparent evidence
ADJUSTMENT_RULES: List[Dict[str, Any]] = [
    {
        "name": "Labor-intensive / public works",
        "pattern": re.compile(r"\b(labor[\-\s]?intensive|public works|cash[\-\s]?for[\-\s]?work)\b", re.I),
        "jobs_per_million_mult": 1.4,
        "direct_share_delta": +0.05,
        "better_job_delta_direct": -0.04,   # often temporary; quality may be lower unless standards present
        "better_job_delta_indirect": -0.02,
        "quality_note": "Short-term, low-wage unless paired with standards/skills."
    },
    {
        "name": "MSME finance / credit line",
        "pattern": re.compile(r"\b(msme|sme|credit line|matching grant|partial credit guarantee)\b", re.I),
        "jobs_per_million_mult": 1.15,
        "direct_share_delta": -0.08,
        "better_job_delta_direct": +0.03,   # formalization/productivity gains possible for beneficiaries
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

# Base better-jobs probability priors (by sector), later adjusted by rules and evidence strength
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

def detect_sectors_weighted(pages: List[str]) -> Tuple[List[Tuple[str, float]], List[Dict[str, Any]]]:
    """
    Returns a list of (sector, weight) summing to 1.0 for top 2 sectors (if available),
    plus evidence items [{sector, page, quote}].
    """
    scores: Dict[str, int] = {}
    evidence = []
    for i, page in enumerate(pages):
        low = page.lower()
        for sector, cues in SECTOR_CUES.items():
            for cue in cues:
                if cue and cue in low:
                    cnt = low.count(cue)
                    scores[sector] = scores.get(sector, 0) + cnt
                    if cnt > 0:
                        # capture one example per sector
                        if not any(ev.get("sector") == sector for ev in evidence):
                            m = re.search(re.escape(cue), low, re.I)
                            if m:
                                evidence.append({"sector": sector, "page": i + 1, "quote": exact_sentence(page, m.span())})

    if not scores:
        return [("Other / General", 1.0)], evidence

    # normalize top 2
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:2]
    total = sum(s for _, s in top)
    weighted = [(sec, s / total if total > 0 else 0.5) for sec, s in top]
    return weighted, evidence

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
                    "better_job_delta_direct": rule["better_job_delta_direct"],
                    "better_job_delta_indirect": rule["better_job_delta_indirect"],
                    "quality_note": rule["quality_note"],
                    "page": i + 1,
                    "quote": sent
                })
    # deduplicate
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
def compute_uncertainty_scale(amount_conf: str, sector_evidence_count: int, n_rules: int) -> Dict[str, float]:
    """
    Returns dispersion controls for Monte Carlo. Lower sigma => tighter (reduced) uncertainty.
    """
    # base dispersions
    sigma_jobs = 0.45   # lognormal sigma for jobs_per_million (base)
    k_direct = (15, 20) # Beta parameters base for direct share

    # amount confidence effect
    if amount_conf.startswith("High"):
        sigma_jobs -= 0.12
    elif amount_conf.startswith("Medium"):
        sigma_jobs -= 0.04
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
        sigma_jobs -= 0.05
    elif n_rules >= 1:
        sigma_jobs -= 0.02
    else:
        sigma_jobs += 0.04

    # clamp
    sigma_jobs = float(np.clip(sigma_jobs, 0.12, 0.60))

    # Direct share concentration via Beta prior strength
    # More evidence => stronger (less variance)
    base_alpha, base_beta = k_direct
    strength_delta = 10 if n_rules >= 3 else (6 if n_rules >= 1 else 0)
    if amount_conf.startswith("High"):
        strength_delta += 6
    if sector_evidence_count >= 2:
        strength_delta += 6
    alpha = base_alpha + strength_delta
    beta = base_beta + strength_delta

    return {"sigma_jobs": sigma_jobs, "alpha_direct": alpha, "beta_direct": beta}

def ai_estimate_jobs_and_better(pages: List[str]) -> Dict[str, Any]:
    full_text = "\n".join(pages)

    # 1) Sector detection (weighted)
    sector_weights, sector_evs = detect_sectors_weighted(pages)
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

    # 4) Adjustment rules from PAD text
    adj_hits = detect_adjustments(pages)
    jobs_per_million = base_jobs_per_million
    direct_share = base_direct_share
    better_p_direct = base_better_prior
    better_p_indirect = base_better_prior - 0.03  # often slightly lower upstream in supply chains

    for h in adj_hits:
        jobs_per_million *= h["jobs_per_million_mult"]
        direct_share = float(np.clip(direct_share + h["direct_share_delta"], 0.05, 0.90))
        better_p_direct = float(np.clip(better_p_direct + h["better_job_delta_direct"], 0.05, 0.95))
        better_p_indirect = float(np.clip(better_p_indirect + h["better_job_delta_indirect"], 0.05, 0.95))

    # 5) Evidence-driven uncertainty controls
    scales = compute_uncertainty_scale(amount_conf, len(sector_evs), len(adj_hits))
    sigma_jobs = scales["sigma_jobs"]
    alpha_direct = scales["alpha_direct"]
    beta_direct = scales["beta_direct"]

    # 6) Monte Carlo simulation (reduced uncertainty with stronger evidence)
    N = 5000
    # Lognormal for jobs_per_million (mean at current point)
    # We convert mean -> mu for lognormal via mean = exp(mu + 0.5*sigma^2)
    mu = np.log(max(jobs_per_million, 1e-6)) - 0.5 * sigma_jobs**2
    draws_jpm = np.random.lognormal(mean=mu, sigma=sigma_jobs, size=N)

    # Beta for direct share (centered at current point by scaling alpha/beta)
    # We adjust alpha/beta to match our current mean while keeping total strength = alpha+beta
    total_strength = alpha_direct + beta_direct
    mean_target = float(np.clip(direct_share, 0.05, 0.95))
    alpha_adj = mean_target * total_strength
    beta_adj = (1 - mean_target) * total_strength
    draws_direct_share = np.random.beta(alpha_adj, beta_adj, size=N)

    # Slight uncertainty around better job probabilities
    bj_sigma = 0.08 - min(0.05, 0.01 * (len(adj_hits) + len(sector_evs)))  # narrower with more evidence
    bj_sigma = float(np.clip(bj_sigma, 0.02, 0.08))
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

    # Medians and P10-P90
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
            } for h in adj_hits]
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
            "note": "Estimates derived from sector weights, financing, and PAD signals; see adjustment quotes."
        }
    }

    # Express a compact ±% uncertainty around medians for display
    med = out["distributions"]["total_jobs"]["p50"]
    halfspan = (out["distributions"]["total_jobs"]["p90"] - out["distributions"]["total_jobs"]["p10"]) / 2.0
    plus_minus_pct = int(np.clip(100.0 * halfspan / med if med > 0 else 50.0, 8, 45))
    out["uncertainty_pct"] = plus_minus_pct
    return out


# ============================================================
# Better Jobs estimate for PAD-explicit counts
# ============================================================
def estimate_better_from_explicit(direct: Optional[int], indirect: Optional[int], pages: List[str]) -> Dict[str, Any]:
    # Reuse sector weights & signals to compute better-job probabilities
    sector_weights, sector_evs = detect_sectors_weighted(pages)
    adj_hits = detect_adjustments(pages)

    base_better_prior = 0.0
    for sec, w in sector_weights:
        base_better_prior += w * BETTER_JOBS_PRIOR.get(sec, BETTER_JOBS_PRIOR["Other / General"])

    better_p_direct = base_better_prior
    better_p_indirect = base_better_prior - 0.03
    for h in adj_hits:
        better_p_direct = float(np.clip(better_p_direct + h["better_job_delta_direct"], 0.02, 0.98))
        better_p_indirect = float(np.clip(better_p_indirect + h["better_job_delta_indirect"], 0.02, 0.98))

    # Slight uncertainty around better-job probabilities (narrow if many cues)
    bj_sigma = 0.06 - min(0.04, 0.01 * (len(adj_hits) + len(sector_evs)))
    bj_sigma = float(np.clip(bj_sigma, 0.015, 0.06))
    N = 4000
    draws_better_direct = np.clip(np.random.normal(better_p_direct, bj_sigma, size=N), 0.02, 0.98)
    draws_better_indirect = np.clip(np.random.normal(better_p_indirect, bj_sigma, size=N), 0.02, 0.98)

    def pct(arr, q): return float(np.percentile(arr, q))

    out = {"evidence": {"sector_quotes": sector_evs or None, "adjustment_quotes": adj_hits or None}}
    if direct is not None:
        bd = np.round(direct * draws_better_direct)
        out["better_direct"] = {"p10": pct(bd, 10), "p50": pct(bd, 50), "p90": pct(bd, 90), "p": better_p_direct}
    if indirect is not None:
        bi = np.round(indirect * draws_better_indirect)
        out["better_indirect"] = {"p10": pct(bi, 10), "p50": pct(bi, 50), "p90": pct(bi, 90), "p": better_p_indirect}
    return out


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
        # ------------------------------ EXPLICIT MODE ------------------------------
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
        # ------------------------------ AI FALLBACK MODE ------------------------------
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

        # Evidence & Assumptions
        st.subheader("PAD Evidence and Signals Used")
        if ai["evidence"].get("amount_quote"):
            aq = ai["evidence"]["amount_quote"]
            if aq.get("page"):
                st.markdown(f"**Financing amount** — page {aq['page']}")
            else:
                st.markdown("**Financing amount**")
            st.markdown(f"> {aq['quote']}")

        if ai["evidence"].get("sector_quotes"):
            st.markdown("**Sector cues (examples):**")
            for s in ai["evidence"]["sector_quotes"]:
                st.markdown(f"- **{s['sector']}** — page {s['page']}")
                st.markdown(f"  > {s['quote']}")

        adj = ai["assumptions"]["adjustment_rules_fired"]
        if adj:
            st.markdown("**Signals that shaped quantity & quality (with quotes):**")
            for h in adj:
                st.markdown(
                    f"- **{h['name']}** (page {h['page']}) — multiplier={h['jobs_per_million_mult']} | "
                    f"Δdirect_share={h['direct_share_delta']:+.02f} | Better Δ (dir/ind) "
                    f"= ({h['better_job_delta_direct']:+.02f}/{h['better_job_delta_indirect']:+.02f})"
                )
                st.markdown(f"  _{h['quality_note']}_")
                st.markdown(f"  > {h['quote']}")
        else:
            st.markdown("_No special adjustment rules triggered from PAD text._")

        with st.expander("What counts as a 'Better Job' here?"):
            st.markdown(
                "- **Higher wages/productivity potential** (skills, certification, upgrading)\n"
                "- **Formality & social protection** (contracts, benefits, compliance)\n"
                "- **Workplace safety (OSH)** (procedures, PPE, inspections)\n"
                "- **Inclusion** (women, disability, childcare—reduces barriers)\n"
                "- **Environmental quality** (green/clean production)\n\n"
                "The model reads PAD signals on these aspects and estimates the probability that a job is 'better'.\n"
                "Monte Carlo simulation yields **P50** and **P10–P90** ranges. Uncertainty narrows as evidence strengthens."
            )

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
