import re
import spacy
from transformers import pipeline
from docx import Document
import fitz  # PyMuPDF

# Load NLP models
summarizer = pipeline("summarization")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load or download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_job_related_sections(text):
    job_keywords = ['employment', 'jobs', 'labor', 'workforce', 'job creation', 'skills', 'training']
    paragraphs = text.split('\n\n')
    return " ".join([p for p in paragraphs if any(k in p.lower() for k in job_keywords)])

def estimate_jobs(text):
    direct = re.findall(r'(\d{3,})\s+(direct|permanent)?\s*jobs?', text, re.IGNORECASE)
    indirect = re.findall(r'(\d{3,})\s+(indirect|temporary)?\s*jobs?', text, re.IGNORECASE)
    return {
        "direct_jobs_estimate": sum(int(j[0]) for j in direct),
        "indirect_jobs_estimate": sum(int(j[0]) for j in indirect)
    }

def assess_job_quality(text):
    labels = ["formal employment", "skills training", "gender inclusion", "youth employment", "decent work"]
    result = classifier(text, candidate_labels=labels)
    return dict(zip(result["labels"], [round(s, 2) for s in result["scores"]]))

def analyze_pad(file):
    full_text = extract_text(file)
    job_text = extract_job_related_sections(full_text)
    job_estimates = estimate_jobs(job_text)
    job_quality = assess_job_quality(job_text)
    summary = summarizer(job_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return {
        "summary": summary,
        "job_estimates": job_estimates,
        "job_quality_indicators": job_quality
    }
