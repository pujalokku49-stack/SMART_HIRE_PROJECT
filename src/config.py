import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
JOB_TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "job_tfidf_vectorizer.pkl")
JOB_VECTORS_PATH = os.path.join(DATA_PROCESSED_DIR, "job_vectors.pkl")
JOB_CORPUS_PATH = os.path.join(DATA_PROCESSED_DIR, "job_corpus_final_v2.csv")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")

GENERIC_WORDS = {
    "experience", "work", "team", "skills", "role", "profile", "position", "job",
    "description", "required", "company", "industry", "ability", "including",
    "time", "date", "business", "management", "customer", "service", "support",
    "staff", "senior", "timely", "related"
}