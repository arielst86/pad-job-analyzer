import streamlit as st
import PyPDF2
import re
import matplotlib.pyplot as plt
import json

# --- Job Estimation Logic ---
def estimate_jobs(text, jobs_per_million=10, direct_pct=0.6, indirect_pct=0.4, custom_multiplier=None, explanation_level="detailed"):
    text = text.lower()

    # --- Sector detection ---
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

    # --- Investment detection ---
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

    # --- Job calculations ---
    base_jobs = amount * jobs_per_million
    direct_jobs = int(base_jobs * direct_pct * multiplier)
    indirect_jobs = int(base_jobs * indirect_pct * multiplier)

    # --- Better/More Jobs Detection ---
    better_keywords = ["skills", "training", "capacity building", "labor standards"]
    more_keywords = ["job creation", "employment opportunities", "labor demand", "msmes"]

    better_jobs = any(k in text for k in better_keywords)
    more_jobs = any(k in text for k in more_keywords)

    # --- Explanations ---
    if explanation_level == "short":
        direct_explanation = f"{direct_jobs} direct jobs estimated based on the investment and sector."
        indirect_explanation = f"{indirect_jobs} indirect jobs estimated from supporting activities."
    else:
        direct_explanation = (
            f"Based on an investment of approximately ${amount:.2f} million and the sector identified as '{sector}', "
            f"we estimate {direct_jobs} direct jobs. This is calculated using a base rate of {jobs_per_million} jobs per million USD, "
            f"adjusted by a sector multiplier of {multiplier}. About {int(direct_pct * 100)}% of total jobs are assumed to be direct, "
            f"including roles like construction, engineering, and project implementation staff."
        )
        indirect_explanation = (
            f"An estimated {indirect_jobs} indirect jobs are expected as a result of the same investment. "
            f"These jobs arise in supporting industries such as supply chains, logistics, and services. "
            f"The {int(indirect_pct * 100)}% share reflects typical indirect job creation patterns in development projects, "
            f"also adjusted by the sector multiplier of {multiplier}."
        )

    return {
        "sector": sector,
        "investment_estimate_million_usd": amount,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "direct_explanation": direct_explanation,
        "indirect_explanation": indirect_explanation,
        "investment_sentence": investment_sentence,
        "sector_sentence": sector_sentence,
        "confidence": confidence,
        "better_jobs": better_jobs,
        "more_jobs": more_jobs
    }

# --- Streamlit UI ---
st.set_page_config(page_title="PAD Job Analyzer", layout="wide")
st.title("📄 PAD Job Creation Analyzer")

uploaded_file = st.file_uploader("Upload a PAD PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    st.subheader("⚙️ Customize Assumptions")
    jobs_per_million = st.slider("Jobs per $1M investment", 1, 50, 10)
    direct_pct = st.slider("Direct job share (%)", 10, 90, 60) / 100
    indirect_pct = 1 - direct_pct
    custom_multiplier = st.number_input("Custom sector multiplier (optional)", min_value=0.1, max_value=5.0, value=1.0)
    use_custom_multiplier = st.checkbox("Use custom multiplier?", value=False)
    explanation_level = st.radio("Explanation detail", ["short", "detailed"], index=1)

    st.subheader("📊 Job Creation Estimate")
    results = estimate_jobs(
        full_text,
        jobs_per_million=jobs_per_million,
        direct_pct=direct_pct,
        indirect_pct=indirect_pct,
        custom_multiplier=custom_multiplier if use_custom_multiplier else None,
        explanation_level=explanation_level
    )

    st.markdown(f"**Sector:** {results['sector'].capitalize()}")
    st.markdown(f"**Investment Estimate:** ${results['investment_estimate_million_usd']:.2f} million")
    st.markdown(f"**Confidence Level:** {results['confidence']}")

    st.markdown(f"**Direct Jobs:** {results['direct_jobs']}")
    st.info(results["direct_explanation"])

    st.markdown(f"**Indirect Jobs:** {results['indirect_jobs']}")
    st.info(results["indirect_explanation"])

    # Tags
    tags = []
    if results["better_jobs"]:
        tags.append("🟢 Better Jobs (skills, training, labor standards)")
    if results["more_jobs"]:
        tags.append("🔵 More Jobs (job creation, MSMEs, labor demand)")
    if tags:
        st.markdown("### 🏷️ Additional Job Dimensions")
        for tag in tags:
            st.success(tag)

    # Source evidence
    st.markdown("### 📌 Source Evidence")
    st.markdown(f"**Investment Reference:** _{results['investment_sentence'].strip()}_")
    if results["sector_sentence"]:
        st.markdown(f"**Sector Reference:** _{results['sector_sentence'].strip()}_")

    # Visualization
    st.markdown("### 📈 Job Distribution")
    fig, ax = plt.subplots()
    ax.pie(
        [results["direct_jobs"], results["indirect_jobs"]],
        labels=["Direct Jobs", "Indirect Jobs"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#2196F3"]
    )
    ax.axis("equal")
    st.pyplot(fig)

    # Download
    st.markdown("### 💾 Download Results")
    json_data = json.dumps(results, indent=2)
    st.download_button("Download as JSON", data=json_data, file_name="job_estimate.json", mime="application/json")
