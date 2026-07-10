import re
import pandas as pd


def fix_resume_encoding(text):
    """Reverse double UTF-8/Latin-1 encoding corruption found in the resume dataset."""
    try:
        return text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text

def clean_resumes(resumes):
    """Fix encoding, drop duplicates, drop empty rows, and known bad category labels."""
    resumes = resumes.copy()
    resumes["Resume"] = resumes["Resume"].apply(fix_resume_encoding)
    resumes = resumes.dropna(subset=["Resume", "Category"])
    resumes = resumes.drop_duplicates().reset_index(drop=True)

    bad_categories = ["Java Developer", "DevOps Engineer", "Testing"]
    resumes = resumes[~resumes["Category"].isin(bad_categories)].reset_index(drop=True)
    return resumes


def build_job_corpus(naukri, linkedin):
    """Standardize naukri + linkedin into one merged job corpus with a common schema."""
    naukri_clean = naukri.rename(columns={"job-description": "description"})
    naukri_clean = naukri_clean[
        ["title", "company", "location", "skills", "description", "experience"]
    ].copy()
    naukri_clean["skills"] = naukri_clean["skills"].fillna("")
    naukri_clean["source"] = "naukri"

    linkedin_clean = linkedin.copy()
    linkedin_clean["company"] = ""
    linkedin_clean["location"] = ""
    linkedin_clean["experience"] = ""
    linkedin_clean = linkedin_clean[
        ["title", "company", "location", "skills", "description", "experience"]
    ]
    linkedin_clean["source"] = "linkedin"

    job_corpus = pd.concat([naukri_clean, linkedin_clean], ignore_index=True)
    return job_corpus


def clean_text(text):
    """Lowercase, strip punctuation/special chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text