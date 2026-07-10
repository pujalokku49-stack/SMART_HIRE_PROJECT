import re
import numpy as np


def extract_years_of_experience(text):
    """Pull a rough years-of-experience number out of free text (resume or job posting)."""
    if not isinstance(text, str):
        return 0
    matches = re.findall(r"(\d+)\+?\s*(?:years|yrs|yr)", text.lower())
    if matches:
        return max(int(m) for m in matches)
    return 0


def parse_job_experience_range(exp_field):
    """Naukri 'experience' field looks like '2-5 Yrs' or '0-1 Yrs'. Returns (min, max)."""
    if not isinstance(exp_field, str):
        return (0, 0)
    matches = re.findall(r"\d+", exp_field)
    if len(matches) >= 2:
        return (int(matches[0]), int(matches[1]))
    elif len(matches) == 1:
        return (int(matches[0]), int(matches[0]))
    return (0, 0)


EDUCATION_KEYWORDS = {
    "bachelor", "b.tech", "btech", "bsc", "b.sc", "ba", "bcom",
    "master", "m.tech", "mtech", "msc", "m.sc", "mba", "phd", "doctorate"
}


def education_overlap(resume_text, job_text):
    """Boolean: does the resume and the job mention at least one shared education keyword?"""
    resume_words = set(str(resume_text).lower().split())
    job_words = set(str(job_text).lower().split())
    resume_edu = resume_words & EDUCATION_KEYWORDS
    job_edu = job_words & EDUCATION_KEYWORDS
    return int(len(resume_edu & job_edu) > 0)


def skill_overlap_features(resume_text, job_skills_text):
    """Count and ratio of overlapping tokens between resume text and a job's skills field."""
    resume_words = set(str(resume_text).lower().split())
    job_words = set(str(job_skills_text).lower().split())
    if not job_words:
        return 0, 0.0
    overlap = resume_words & job_words
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / len(job_words)
    return overlap_count, overlap_ratio


def build_feature_row(resume_text, job_row, text_similarity):
    """
    Assemble one feature vector for a (resume, job) pair.
    `text_similarity` is the precomputed TF-IDF cosine similarity between the two.
    """
    resume_years = extract_years_of_experience(resume_text)
    job_min_exp, job_max_exp = parse_job_experience_range(job_row.get("experience", ""))
    experience_match = int(job_min_exp <= resume_years <= job_max_exp) if job_max_exp > 0 else 0
    experience_gap = resume_years - job_max_exp if job_max_exp > 0 else 0

    skill_overlap_count, skill_overlap_ratio = skill_overlap_features(
        resume_text, job_row.get("skills", "")
    )
    edu_match = education_overlap(resume_text, job_row.get("description", ""))

    return {
        "text_similarity": text_similarity,
        "skill_overlap_count": skill_overlap_count,
        "skill_overlap_ratio": skill_overlap_ratio,
        "resume_years_experience": resume_years,
        "job_min_experience": job_min_exp,
        "job_max_experience": job_max_exp,
        "experience_match": experience_match,
        "experience_gap": experience_gap,
        "education_match": edu_match,
        "resume_length": len(str(resume_text).split()),
    }