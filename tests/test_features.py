import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from features.text_features import clean_text
from data.preprocess import clean_resumes
from features.match_features import (
    extract_years_of_experience,
    parse_job_experience_range,
    skill_overlap_features,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World! 123") == "hello world 123"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("too    many   spaces") == "too many spaces"


def test_clean_resumes_drops_bad_categories():
    df = pd.DataFrame({
        "Resume": ["text a", "text b", "text c"],
        "Category": ["Java Developer", "Data Science", "Testing"]
    })
    result = clean_resumes(df)
    assert "Java Developer" not in result["Category"].values
    assert "Testing" not in result["Category"].values
    assert "Data Science" in result["Category"].values


def test_extract_years_of_experience():
    assert extract_years_of_experience("5 years of experience in Python") == 5
    assert extract_years_of_experience("no mention here") == 0


def test_parse_job_experience_range():
    assert parse_job_experience_range("2-5 Yrs") == (2, 5)
    assert parse_job_experience_range("3 Yrs") == (3, 3)
    assert parse_job_experience_range(None) == (0, 0)


def test_skill_overlap_features():
    count, ratio = skill_overlap_features("python sql excel", "python java")
    assert count == 1
    assert 0 < ratio <= 1