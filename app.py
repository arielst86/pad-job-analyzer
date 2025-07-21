import streamlit as st
import PyPDF2
import re
import pandas as pd
import requests
from io import BytesIO

# --- Query World Bank Projects API for Equivalent Projects ---
def get_equivalent_projects(sector, amount_million):
    url = f"https://search.worldbank.org/api/v2/projects?q=sectorname:{sector}&format=json"
    try:
        response = requests.get(url)
        data = response.json()
        projects = []
        for pid, project in data.get("projects", {}).items():
            try:
                status = project.get("projectstatus", "").lower()
                net_commitment = float(project.get("totalcommamt", 0)) / 1_000_000
                if status == "closed" and abs(net_commitment - amount_million) / amount_million <= 0.25:
                    rf = project.get("resultsframework", "")
                    job_match = re.search(r"(\d{1,3}(?:,\d{3})*)\s+jobs", rf.lower()) if rf else None
                    jobs_created = int(job_match.group(1).replace(",", "")) if job_match else "N/A"
                    projects.append({
                        "Project Name": project.get("project_name", "Unnamed"),
                        "Country": project.get("countryshortname", "Unknown"),
                        "P-Code": pid,
                        "Dates": f"{project.get('boardapprovaldate', '')[:10]} to {project.get('closingdate', '')[:10]}",
                        "Jobs Created": jobs_created,
                        "Similarity": "Same sector, similar commitment, closed"
                    })
            except:
                continue
        return projects
    except Exception as e:
        print("API error:", e)
        return []

# --- Job Estimation Logic ---
def estimate_jobs(text, jobs_per_million=10, direct_pct=0.6, indirect_pct=0.4):
    text = text.lower()

    sector_weights = {
        "infrastructure": 1.5,
        "education": 1.2,
        "health": 1.3,
        "agriculture": 1.4,
        "digital": 1.1
    }
    sector = "general"
    sector_sentence = ""
    for s in sector_weights:
        if s in text:
            sector = s
            match = re.search(rf"([^.]*\b{s}\b[^.]*)", text)
            if match:
                sector_sentence = match.group(1)
            break
    multiplier = sector_weights.get(sector, 1.0)

    match = re.search(r"(?:\$)?\s?([\d,]+\.?\d*)\s?(million|billion)", text)
    amount = None
    investment_sentence = ""
    if match:
        try:
            amount = float(match.group(1).replace(",", ""))
            if match.group(2).lower() == "billion":
                amount *= 1000
            investment_sentence = match.group(0)
        except:
            pass
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
    better_jobs_flag = any(k in text for k in better_keywords)
    better_direct_jobs = int(direct_jobs * 0.3) if better_jobs_flag else 0
    better_indirect_jobs = int(indirect_jobs * 0.2) if better_jobs_flag else 0

    quote_match = re.search(r"([^.]*?(?:job creation|employment|labor|msmes|skills|training)[^.]*\.)", text)
    source_quote = quote_match.group(1).strip() if quote_match else "No specific quote found."

    equivalent_projects = get_equivalent_projects(sector, amount)

    return {
        "sector": sector,
        "investment_estimate_million_usd": amount,
        "direct_jobs": direct_jobs,
        "indirect_jobs": indirect_jobs,
        "better_direct_jobs": better_direct_jobs,
        "better_indirect_jobs": better_indirect_jobs,
        "investment_sentence": investment_sentence,
        "sector_sentence": sector_sentence,
        "confidence": confidence,
        "source_quote": source_quote,
        "equivalent_projects": equivalent_projects
    }

# --- Streamlit UI ---
st.set_page_config(page_title="PAD Job Analyzer", layout="wide")

# --- Header ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size: 2.5em; font-weight: bold; color: #003366;">PAD Job Creation Analyzer</div>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/World_Bank_logo.svg/512px-World_Bank_logo.svg.png" height="
    unsafe_allow_html=True
)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PAD PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    results = estimate_jobs(full_text)

    st.subheader("Job Creation Estimate")
    st.markdown(f"**Sector:** {results['sector'].capitalize()}")
    st.markdown(f"**Investment Estimate:** ${results['investment_estimate_million_usd']:.2f} million")
    st.markdown(f"**Confidence Level:** {results['confidence']}")

    st.markdown(f"**Direct Jobs:** {results['direct_jobs']}")
    st.info(
        f"Based on an investment of approximately ${results['investment_estimate_million_usd']:.2f} million and the sector identified as '{results['sector']}', "
        f"we estimate {results['direct_jobs']} direct jobs. This is calculated using a base rate of 10 jobs per million USD, "
        f"adjusted by a sector multiplier. This estimate is consistent with findings from ILO and World Bank literature on employment elasticity in development projects."
    )
    st.markdown(f"**Estimated Better Direct Jobs:** {results['better_direct_jobs']}")
    st.info(
        f"Approximately {results['better_direct_jobs']} of the direct jobs are considered 'better jobs'. This is based on the presence of keywords such as "
        f"'skills', 'training', 'capacity building', and 'labor standards' in the PAD. These indicators suggest that a portion of the direct employment will involve "
        f"improved working conditions, upskilling, or formal labor protections, aligning with the World Bank’s Better Jobs agenda."
    )

    st.markdown(f"**Indirect Jobs:** {results['indirect_jobs']}")
    st.info(
        f"An estimated {results['indirect_jobs']} indirect jobs are expected as a result of the same investment. These jobs arise in supporting industries such as "
        f"supply chains, logistics, and services. This estimate is informed by multiplier effects observed in World Bank infrastructure and agriculture projects."
    )
    st.markdown(f"**Estimated Better Indirect Jobs:** {results['better_indirect_jobs']}")
    st.info(
        f"Roughly {results['better_indirect_jobs']} of the indirect jobs are also expected to be 'better jobs'. This reflects the likelihood that improved labor standards "
        f"and training components in the project will extend to subcontractors, suppliers, and service providers."
    )

    st.subheader("Source Evidence")
    st.markdown(f"**Investment Reference:** *{results['investment_sentence']}*")
    if results["sector_sentence"]:
        st.markdown(f"**Sector Reference:** *{results['sector_sentence']}*")
    st.markdown(f"**Quoted Source Text:**\n> {results['source_quote']}")

    # --- Equivalent Projects Table ---
    if results["equivalent_projects"]:
        st.subheader("Equivalent Projects")
        eq_df = pd.DataFrame(results["equivalent_projects"])
        st.dataframe(eq_df)

    st.subheader("Download Results")
    df = pd.DataFrame([results])
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Job Estimate')
    st.download_button("Download as Excel", data=output.getvalue(), file_name="job_estimate.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
