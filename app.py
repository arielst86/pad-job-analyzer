import streamlit as st
import PyPDF2
import re
from collections import Counter

# --- Job Estimation Logic ---
def estimate_jobs(text):
    text = text.lower()

    keywords = {
        "direct": ["direct employment", "hiring", "staffing", "recruitment"],
        "indirect": ["indirect jobs", "supply chain", "msmes", "contractors"],
        "better": ["skills", "training", "capacity building", "labor standards"],
        "more": ["job creation", "employment opportunities", "labor demand"]
    }

    counts = {k: sum(word in text for word in v) for k, v in keywords.items()}

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

    explanation = (
        f"The estimate is based on a detected investment of approximately ${amount:.2f} million. "
        f"A base assumption of 10 jobs per million USD is applied, adjusted by a sector multiplier "
        f"of {multiplier} for the '{sector}' sector. "
        f"60% of the jobs are considered direct (e.g., construction, staffing), and 40% indirect "
        f"(e.g., supply chain, services)."
    )

    return {
        "sector": sector,
        "investment_estimate_million_usd": amount,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "keyword_counts": counts,
        "explanation": explanation
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
    st.json({
        "Sector": results["sector"],
        "Investment (Million USD)": results["investment_estimate_million_usd"],
        "Direct Jobs": results["direct_jobs"],
        "Indirect Jobs": results["indirect_jobs"],
        "Keyword Counts": results["keyword_counts"]
    })

    st.markdown("### 🧠 Estimation Logic")
    st.info(results["explanation"])
