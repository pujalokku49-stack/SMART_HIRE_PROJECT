import pandas as pd
import os


def load_raw_datasets(data_raw_dir):
    """Load the three raw datasets from data/raw/."""
    resumes = pd.read_csv(
        os.path.join(data_raw_dir, "resumes", "resumes.csv"), encoding="utf-8"
    )
    naukri = pd.read_csv(os.path.join(data_raw_dir, "naukri", "naukri.csv"))
    linkedin = pd.read_csv(os.path.join(data_raw_dir, "linkedin", "linkedin.csv"))
    return resumes, naukri, linkedin