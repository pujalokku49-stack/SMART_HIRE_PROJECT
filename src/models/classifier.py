import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.text_features import clean_text
import config


class ResumeClassifier:
    def __init__(self):
        with open(config.CLASSIFIER_PATH, "rb") as f:
            self.clf = pickle.load(f)
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            self.tfidf = pickle.load(f)

    def predict(self, resume_text):
        cleaned = clean_text(resume_text)
        vector = self.tfidf.transform([cleaned])
        prediction = self.clf.predict(vector)[0]
        probabilities = self.clf.predict_proba(vector)[0]
        confidence = max(probabilities)
        return prediction, confidence