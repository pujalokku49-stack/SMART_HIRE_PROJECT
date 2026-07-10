import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.text_features import clean_text
import config


class JobRecommender:
    def __init__(self):
        with open(config.JOB_TFIDF_VECTORIZER_PATH, "rb") as f:
            self.job_tfidf = pickle.load(f)
        with open(config.JOB_VECTORS_PATH, "rb") as f:
            self.job_vectors = pickle.load(f)
        self.job_corpus = pd.read_csv(config.JOB_CORPUS_PATH)
        self.job_corpus["skills"] = self.job_corpus["skills"].fillna("")

    def recommend(self, resume_text, top_n=10, include_fit_columns=False):
        cleaned = clean_text(resume_text)
        resume_vector = self.job_tfidf.transform([cleaned])
        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]

        display_cols = ["title", "company", "location", "source"]
        if include_fit_columns:
            display_cols += ["skills", "description", "experience"]

        results = self.job_corpus.iloc[top_indices][display_cols].copy()
        results["match_score"] = similarities[top_indices]
        return results.reset_index(drop=True)

    def skill_gap_report(self, resume_text, top_n_jobs=20, top_n_skills=15):
        cleaned = clean_text(resume_text)
        resume_vector = self.job_tfidf.transform([cleaned])
        resume_words = set(cleaned.split())

        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_n_jobs]
        top_jobs = self.job_corpus.iloc[top_indices]

        avg_vector = self.job_vectors[top_indices].mean(axis=0)
        avg_vector = np.asarray(avg_vector).flatten()

        feature_names = self.job_tfidf.get_feature_names_out()
        top_word_indices = avg_vector.argsort()[::-1][:100]

        candidate_words = [feature_names[i] for i in top_word_indices]
        candidate_words = [w for w in candidate_words if w not in config.GENERIC_WORDS and len(w) > 2]

        matched = [w for w in candidate_words if w in resume_words][:top_n_skills]
        missing = [w for w in candidate_words if w not in resume_words][:top_n_skills]

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "based_on_jobs": top_jobs["title"].head(5).tolist(),
        }