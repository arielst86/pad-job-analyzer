import streamlit as st
import PyPDF2
import re

# --- Job Estimation Logic ---
def estimate_jobs(text):
    text = text.lower()

    # Sector-based heuristic
    sector_weights = {
        "infrastructure": 1.5,
        "education": 1.2,
        "health": 1.3,
        "agriculture": 1.4,
        "digital": 1.1
    }

    sector = next((s for s in sector_weights if s in text), "general")
    multiplier = sector_weights.get(sector, 1.0)

    # --- Investment-based heuristic ---
    amount = None

    # Try matching "$300 million", "300 million", etc.
    match = re.search(r"(?:\$)?\s?([\d,]+\.?\d*)\s?(million|billion)", text)
    if match:
        try:
            amount = float(match.group(1).replace(",", ""))
            if match.group(2).lower() == "billion":
                amount *= 1000
        except (ValueError, IndexError):
            amount = None

    # Fallback: match "Total Project Cost" followed by a number
    if amount is None:
        match_alt = re.search(r"total project cost\s*[:\-]?\s*([\d,]+\.?\d*)", text)
        if match_alt:
            try:
                amount = float(match_alt.group(1).replace(",", ""))
            except ValueError:
                amount = None

    # Final fallback
    if amount is None:
        amount = 50
        st.warning("Could not detect investment amount in the text. Using default estimate of $50M.")

    base_jobs = amount * 10
    direct_jobs = int(base_jobs * 0.6 * multiplier)
    indirect_jobs = int(base_jobs * 0.4 * multiplier)

    # Simple explanations
    direct_explanation = (
        f"Based on the document's mention of an investment of approximately ${amount:.2f} million "
        f"in the '{sector}' sector, we estimate {direct_jobs} direct jobs. "
        f"This includes roles like construction, staffing, and operations."
    )

    indirect_explanation = (
        f"In addition, we estimate {indirect_jobs} indirect jobs based on the same investment. "
        f"These include jobs created through supply chains, services, and supporting industries."
    )

    return {
        "sector": sector,
        "investment_estimate_million_usd": amount,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "direct_explanation": direct_explanation,
        "indirect_explanation": indirect_explanation
    }

# --- Streamlit UI ---
st.set_page_config(page_title="PAD Job Analyzer", layout="wide")
st.title("📄 PAD Job Creation Analyzer")

uploaded_file = st.file_uploader("Upload a PAD PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    st.subheader("📘 Extracted Text (Preview)")
    st.text_area("PAD Content", full_text[:3000], height=200)

    st.subheader("📊 Job Creation Estimate")
    results = estimate_jobs(full_text)

    st.markdown(f"**Sector:** {results['sector'].capitalize()}")
    st.markdown(f"**Investment Estimate:** ${results['investment_estimate_million_usd']:.2f} million")

    st.markdown(f"**Direct Jobs:** {results['direct_jobs']}")
    st.info(results["direct_explanation"])

    st.markdown(f"**Indirect Jobs:** {results['indirect_jobs']}")
    st.info(results["indirect_explanation"])
