import streamlit as st
import PyPDF2
import re
import pandas as pd
import requests
import altair as alt
from io import BytesIO

# --- Estimate Jobs from Description ---
def estimate_jobs_created(description):
    keywords = ["employment", "job", "labor", "workforce", "hiring", "recruitment", "livelihood", "skills", "training"]
    score = sum(description.lower().count(k) for k in keywords)
    return score * 100  # Simple multiplier

# --- Fetch Equivalent Projects (Merged Logic) ---
def fetch_equivalent_projects(sector, net_commitment_million, tolerance=0.25):
    base_url = "https://search.worldbank.org/api/v2/projects"
    params = {
        "format": "json",
        "statuscode_exact": "C",  # Closed projects
        "sector_exact": sector,
        "rows": 1000
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        projects = data.get("projects", {}).values()
        similar_projects = []

        for project in projects:
            try:
                commitment = float(project.get("totalcommamt", 0)) / 1_000_000
                if abs(commitment - net_commitment_million) / net_commitment_million <= tolerance:
                    description = project.get("project_abstract", {}).get("cdata", "")
                    similar_projects.append({
                        "Project Name": project.get("project_name", "Unnamed"),
                        "Country": project.get("countryshortname", "Unknown"),
                        "P-Code": project.get("id"),
                        "Dates": f"{project.get('boardapprovaldate', '')[:10]} to {project.get('closingdate', '')[:10]}",
                        "Jobs Created": estimate_jobs_created(description),
                        "Similarity": "Same sector, similar net commitment, closed"
                    })
            except (TypeError, ValueError):
                continue

        return similar_projects
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

    equivalent_projects = fetch_equivalent_projects(sector, amount)

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

st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size: 2.5em; font-weight: bold; color: #003366;">PAD Job Creation Analyzer</div>
        https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/World_Bank_logo.svg/512px-World_Bank_logo.svg.png
    </div>
    """,
    unsafe_allow_html=True
)

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
    st.markdown(f"**Estimated Better Direct Jobs:** {results['better_direct_jobs']}")
    st.markdown(f"**Indirect Jobs:** {results['indirect_jobs']}")
    st.markdown(f"**Estimated Better Indirect Jobs:** {results['better_indirect_jobs']}")

    st.subheader("Source Evidence")
    st.markdown(f"**Investment Reference:** *{results['investment_sentence']}*")
    if results["sector_sentence"]:
        st.markdown(f"**Sector Reference:** *{results['sector_sentence']}*")
    st.markdown(f"**Quoted Source Text:**\n> {results['source_quote']}")

    if results["equivalent_projects"]:
        st.subheader("Equivalent Projects")
        eq_df = pd.DataFrame(results["equivalent_projects"])
        st.dataframe(eq_df)

        # --- Altair Chart ---
        st.subheader("Job Creation in Equivalent Projects")
        chart_data = eq_df[["Project Name", "Jobs Created"]].copy()
        chart_data = chart_data[chart_data["Jobs Created"] != "N/A"]
        chart_data["Jobs Created"] = pd.to_numeric(chart_data["Jobs Created"], errors="coerce")
        chart_data = chart_data.dropna()

        if not chart_data.empty:
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Jobs Created:Q", title="Estimated Jobs Created"),
                y=alt.Y("Project Name:N", sort='-x', title="Project"),
                tooltip=["Project Name", "Jobs Created"]
            ).properties(
                width=700,
                height=400,
                title="Estimated Jobs Created in Equivalent Projects"
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No job creation data available for charting.")

    st.subheader("Download Results")
    df = pd.DataFrame([results])
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Job Estimate')
    st.download_button("Download as Excel", data=output.getvalue(), file_name="job_estimate.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Real-Time Projects Section ---
# --- Real-Time Projects Section ---
def fetch_live_projects(sector):
    """
    Fetch live (active) projects from the World Bank API for the given sector.
    """
    base_url = "https://search.worldbank.org/api/v2/projects"
    params = {
        "format": "json",
        "statuscode_exact": "A",  # Active projects
        "sectorname": sector,
        "rows": 100
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        projects = data.get("projects", {}).values()
        live_projects = []

        for project in projects:
            try:
                commitment_str = str(project.get("totalcommamt", "0")).replace(",", "")
                commitment = float(commitment_str) / 1_000_000
                live_projects.append({
                    "Project Name": project.get("project_name", "Unnamed"),
                    "Country": project.get("countryshortname", "Unknown"),
                    "P-Code": project.get("id", ""),
                    "Dates": f"{project.get('boardapprovaldate', '')[:10]} to {project.get('closingdate', '')[:10]}",
                    "Total Commitment (USD)": commitment
                })
            except ValueError:
                continue  # Skip projects with invalid commitment values

        return pd.DataFrame(live_projects)

    except Exception as e:
        st.error(f"Failed to fetch live projects: {e}")
        return pd.DataFrame()

# --- Display Live Projects ---
def fetch_live_projects(sector):
    """
    Fetch live (active) projects from the World Bank API for the given sector,
    filtering for those that mention jobs in the results framework.
    """
    import re

    base_url = "https://search.worldbank.org/api/v2/projects"
    params = {
        "format": "json",
        "statuscode_exact": "A",  # Active projects
        "sectorname": sector,
        "rows": 100
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        projects = data.get("projects", {}).values()
        live_projects = []

        for project in projects:
            rf = project.get("resultsframework", "")
            if rf:
                job_match = re.search(r"(\d{1,3}(?:,\d{3})*)\s+(?:jobs|employment|positions)", rf.lower())
                if job_match:
                    try:
                        commitment_str = str(project.get("totalcommamt", "0")).replace(",", "")
                        commitment = float(commitment_str) / 1_000_000
                        jobs_mentioned = int(job_match.group(1).replace(",", ""))
                        live_projects.append({
                            "Project Name": project.get("project_name", "Unnamed"),
                            "Country": project.get("countryshortname", "Unknown"),
                            "P-Code": project.get("id", ""),
                            "Dates": f"{project.get('boardapprovaldate', '')[:10]} to {project.get('closingdate', '')[:10]}",
                            "Total Commitment (USD)": commitment,
                            "Jobs Mentioned": jobs_mentioned
                        })
                    except ValueError:
                        continue  # Skip if commitment or job number can't be parsed

        return pd.DataFrame(live_projects)

    except Exception as e:
        st.error(f"Failed to fetch live projects: {e}")
        return pd.DataFrame()
