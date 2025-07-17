import streamlit as st
from pad_analyzer import analyze_pad

st.set_page_config(page_title="PAD Job Analyzer", layout="wide")
st.title("📄 World Bank PAD Job Impact Analyzer")

uploaded_file = st.file_uploader("Upload a PAD file (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    with st.spinner("Analyzing the PAD..."):
        result = analyze_pad(uploaded_file)

    st.success("Analysis complete!")

    st.subheader("🔍 Summary")
    st.write(result["summary"])

    st.subheader("📊 Job Estimates")
    st.json(result["job_estimates"])

    st.subheader("🌱 Job Quality Indicators")
    st.json(result["job_quality_indicators"])
