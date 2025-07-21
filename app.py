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

    # Estimate better jobs as 30% of direct and 20% of indirect if keywords are present
    better_direct_jobs = int(direct_jobs * 0.3) if better_jobs else 0
    better_indirect_jobs = int(indirect_jobs * 0.2) if better_jobs else 0

    return {
        "sector": sector,
        "investment_estimate_million_usd": amount,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "better_direct_jobs": better_direct_jobs,
        "better_indirect_jobs": better_indirect_jobs,
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
        section.main {{
            background-color: {WB_COLORS['background']};
            color: {WB_COLORS['text']};
            font-family: 'Segoe UI', sans-serif;
        }}
        .title-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .title {{
            font-size: 2.5em;
            font-weight: bold;
            color: {WB_COLORS['primary']};
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

# --- Header with Top-Right Logo ---
st.markdown(
    """
    <div class="title-bar">
        <div class="title">PAD Job Creation Analyzer</div>
        <img src="https://upload.wikimedia.org/wikipedia/commons4/World_Bank_logo.svg/512px-World_Bank_logo.svg.png
    </div>
    """,
    unsafe_allow_html=True
)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PAD PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    results = estimate_jobs(full_text)

    st.markdown('<div class="section">Job Creation Estimate</div>', unsafe_allow_html=True)
    st.markdown(f"**Sector:** {results['sector'].capitalize()}")
    st.markdown(f"**Investment Estimate:** ${results['investment_estimate_million_usd']:.2f} million")
    st.markdown(f"**Confidence Level:** {results['confidence']}")

    st.markdown(f"**Direct Jobs:** {results['direct_jobs']}")
    st.markdown(f"<div class='info-box'>{results['direct_explanation']}</div>", unsafe_allow_html=True)

    if results["better_direct_jobs"] > 0:
        st.markdown(
            f"**Estimated Better Direct Jobs:** {results['better_direct_jobs']}")
        st.markdown(
            f"<div class='info-box'>Approximately {results['better_direct_jobs']} of the direct jobs are considered 'better jobs'. "
            f"This estimate is based on the presence of keywords such as 'skills', 'training', 'capacity building', and 'labor standards' in the PAD. "
            f'These indicators suggest that a portion of the direct employment will involve improved working conditions, upskilling, or formal labor protections.</div>',
            unsafe_allow_html=True
        )

    st.markdown(f"**Indirect Jobs:** {results['indirect_jobs']}")
    st.markdown(f"<div class='info-box'>{results['indirect_explanation']}</div>", unsafe_allow_html=True)

    if results["better_indirect_jobs"] > 0:
        st.markdown(
            f"**Estimated Better Indirect Jobs:** {results['better_indirect_jobs']}")
        st.markdown(
            f"<div class='info-box'>Roughly {results['better_indirect_jobs']} of the indirect jobs are also expected to be 'better jobs'. "
            f"This reflects the likelihood that improved labor standards and training components in the project will extend to subcontractors, suppliers, and service providers.</div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="section">Source Evidence</div>', unsafe_allow_html=True)
    st.markdown(f"**Investment Reference:** *{results['investment_sentence'].strip()}*")
    if results["sector_sentence"]:
        st.markdown(f"**Sector Reference:** *{results['sector_sentence'].strip()}*")
    st.markdown(f"**Quoted Source Text:**\n> {results['source_quote']}")

    st.markdown('<div class="section">Download Results</div>', unsafe_allow_html=True)
    df = pd.DataFrame([results])
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Job Estimate')
    st.download_button("Download as Excel", data=output.getvalue(), file_name="job_estimate.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
