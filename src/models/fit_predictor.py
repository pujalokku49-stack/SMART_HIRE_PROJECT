import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from features.text_features import clean_text
from features.match_features import build_feature_row
import config


class FitPredictor:
    def __init__(self):
        with open(config.MODELS_DIR + "/fit_predictor.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(config.JOB_TFIDF_VECTORIZER_PATH, "rb") as f:
            self.job_tfidf = pickle.load(f)

    def predict_fit(self, resume_text, job_row):
        """job_row: a pandas Series/dict with title, skills, description, experience."""
        cleaned = clean_text(resume_text)
        resume_vec = self.job_tfidf.transform([cleaned])
        job_desc_vec = self.job_tfidf.transform([clean_text(job_row.get("description", ""))])
        similarity = cosine_similarity(resume_vec, job_desc_vec)[0][0]

        features = build_feature_row(resume_text, job_row, similarity)
        X = pd.DataFrame([features])

        fit_probability = self.model.predict_proba(X)[0][1]
        return fit_probability