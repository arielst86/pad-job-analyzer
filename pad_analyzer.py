import streamlit as st
import PyPDF2
import re
import pandas as pd
from io import BytesIO

# --- Job Estimation Logic ---
def estimate_jobs(text, jobs_per_million=10, direct_pct=0.6, indirect_pct=0.4, custom_multiplier=None):
    text = text.lower()

    sector_weights = {
        "infrastructure": 1.5,
        "education": 1.2,
        "health": 1.3,
        "agriculture": 1.4,
        "digital": 1.1
    }
    sector_sentence = ""
    sector = "general"
    for s in sector_weights:
        if s in text:
            sector = s
            match = re.search(rf"([^.]*\b{s}\b[^.]*)", text)
            if match:
                sector_sentence = match.group(1)
            break
    multiplier = custom_multiplier if custom_multiplier else sector_weights.get(sector, 1.0)

    amount = None
    investment_sentence = ""
    match = re.search(r"(?:\$)?\s?([\d,]+\.?\d*)\s?(million|billion)", text)
    if match:
        try:
            amount = float(match.group(1).replace(",", ""))
            if match.group(2).lower() == "billion":
                amount *= 1000
            investment_sentence = match.group(0)
        except:
            amount = None

    if amount is None:
        match_alt = re.search(r"(total project cost[^$\n\r]*?([\d,]+\.?\d*))", text)
        if match_alt:
            try:
                amount = float(match_alt.group(2).replace(",", ""))
                investment_sentence = match_alt.group(1)
            except:
                amount = None
    if amount is None:
        amount = 50
        investment_sentence = "No clear investment amount found. Defaulted to $50M."
        confidence = "Low"
    else:
        confidence = "High" if sector != "general" else "Medium"

    base_jobs = amount * jobs_per_million
    direct_jobs = int(base_jobs * direct_pct * multiplier)
    indirect_jobs = int(base_jobs * indirect_pct * multiplier)

    better_keywords = ["skills", "training", "capacity building", "labor standards"]
    more_keywords = ["job creation", "employment opportunities", "labor demand", "msmes"]

    better_jobs = any(k in text for k in better_keywords)
    more_jobs = any(k in text for k in more_keywords)

    quote_match = re.search(r"([^.]*?(?:job creation|employment|labor|msmes|skills|training)[^.]*\.)", text)
    source_quote = quote_match.group(1).strip() if quote_match else "No specific quote found."

    return {
        "sector": sector,
        "investment_estimate_million_usd": amount,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "direct_explanation": (
            f"Based on an investment of approximately ${amount:.2f} million and the sector identified as '{sector}', "
            f"we estimate {direct_jobs} direct jobs. This is calculated using a base rate of {jobs_per_million} jobs per million USD, "
            f"adjusted by a sector multiplier of {multiplier}. About {int(direct_pct * 100)}% of total jobs are assumed to be direct."
        ),
        "indirect_explanation": (
            f"An estimated {indirect_jobs} indirect jobs are expected as a result of the same investment. "
            f"These jobs arise in supporting industries such as supply chains, logistics, and services. "
            f"The {int(indirect_pct * 100)}% share reflects typical indirect job creation patterns in development projects, "
            f"also adjusted by the sector multiplier of {multiplier}."
        ),
        "investment_sentence": investment_sentence,
        "sector_sentence": sector_sentence,
        "confidence": confidence,
        "better_jobs": better_jobs,
        "more_jobs": more_jobs,
        "source_quote": source_quote
    }

# --- Streamlit UI ---
st.set_page_config(page_title="PAD Job Analyzer", layout="wide")

# --- World Bank Color Palette ---
WB_COLORS = {
    "primary": "#003366",
    "accent": "#0072BC",
    "background": "#F2F2F2",
    "highlight": "#E6F2F8",
    "text": "#333333"
}

# --- Custom CSS ---
st.markdown(
    f"""
    <style>
        html, body {{
            background-color: {WB_COLORS['background']};
            color: {WB_COLORS['text']};
            font-family: 'Segoe UI', sans-serif;
        }}
        .title {{
            font-size: 2.5em;
            font-weight: bold;
            color: {WB_COLORS['primary']};
            margin-bottom: 0.5em;
        }}
        .section {{
            font-size: 1.5em;
            color: {WB_COLORS['accent']};
            margin-top: 2em;
            margin-bottom: 0.5em;
        }}
        .info-box {{
            background-color: {WB_COLORS['highlight']};
            padding: 1em;
            border-left: 4px solid {WB_COLORS['accent']};
            margin-bottom: 1em;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/World_Bank_logo.svg/512px-World_Bank_logo.svg.png", width=150)
st.markdown('<div class="title">PAD Job Creation Analyzer</div>', unsafe_allow_html=True)

# --- File Upload ---
uploaded_file