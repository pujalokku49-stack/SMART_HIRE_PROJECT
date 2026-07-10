# SmartHire — Resume-to-Job Matching & Career Guidance Engine

SmartHire is a classical-ML portal that takes an uploaded resume and returns:
1. A predicted career category (supervised classification)
2. A ranked list of the top matching jobs from a merged job corpus (unsupervised content-based recommendation)
3. A fit/shortlisting score for target roles (supervised)
4. A skill-gap report comparing the resume's vocabulary against its best-matching jobs (unsupervised)

No LLMs or generative AI are used anywhere in the pipeline — every prediction comes from TF-IDF vectorization, cosine similarity, Logistic Regression / Random Forest / Gradient Boosting, and K-Means clustering.

## Project Structure

```
SmartHire_Project/
├── app/
│   └── streamlit_app.py        # web portal: upload → category → jobs → fit score → skill gaps
├── notebooks/
│   ├── 01_eda.ipynb            # data cleaning, job corpus merge
│   ├── 02_resume_classifier.ipynb
│   ├── 03_recommender.ipynb
│   ├── 04_clustering_topics.ipynb
│   └── 05_fit_predictor.ipynb
├── src/
│   ├── config.py
│   ├── data/                   # load_data.py, preprocess.py
│   ├── features/                # text_features.py, match_features.py
│   ├── models/                  # classifier.py, recommender.py, fit_predictor.py
│   ├── parsing/                  # resume_parser.py (PDF/DOCX/TXT extraction)
│   └── evaluate.py              # shared evaluation metrics
├── tests/
│   └── test_features.py
├── reports/
│   ├── figures/                 # confusion matrix, elbow plot, cluster plot
│   └── final_report.pdf
├── data/                        # gitignored — see setup below
├── models/                      # gitignored — trained .pkl artifacts
└── requirements.txt
```

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/pujalokku49-stack/SmartHire_Project.git
   cd SmartHire_Project
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. Download the datasets from Kaggle and place them under `data/raw/`:
   - Resume dataset → `data/raw/resumes/resumes.csv`
   - Naukri job postings → `data/raw/naukri/naukri.csv`
   - LinkedIn job postings → `data/raw/linkedin/linkedin.csv`

3. Run the notebooks in order to reproduce the pipeline (each writes intermediate/processed data and saves model artifacts):
   ```
   01_eda.ipynb              → data/interim/job_corpus.csv, data/processed/resumes_clean.csv
   02_resume_classifier.ipynb → models/classifier.pkl, models/tfidf_vectorizer.pkl
   03_recommender.ipynb       → models/job_tfidf_vectorizer.pkl, data/processed/job_vectors.pkl
   04_clustering_topics.ipynb → models/kmeans_model.pkl, data/processed/job_corpus_final_v2.csv
   05_fit_predictor.ipynb     → models/fit_predictor.pkl, models/fit_predictor_scaler.pkl
   ```

4. Run the tests:
   ```bash
   pip install pytest --break-system-packages
   pytest tests/ -v
   ```

5. Launch the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Results Summary

| Component | Metric | Value |
|---|---|---|
| Resume Classifier (Random Forest) | Accuracy | 0.70 |
| Resume Classifier (Logistic Regression) | Accuracy | 0.66 |
| Resume Classifier (Random Forest) | ROC-AUC (macro, OvR) | 0.97 |
| Job Recommender | Precision@10 (category-word proxy) | 0.25 |
| Job Clustering (K-Means, k=10) | Silhouette Score | 0.024 |
| Fit Predictor | Accuracy / ROC-AUC | see notebook 05 output |

See `reports/final_report.pdf` for full methodology, error analysis, and limitations.

## Known Limitations

- The resume dataset's `Category` labels for "Java Developer", "DevOps Engineer", and "Testing" were dropped during cleaning due to inconsistent/overlapping content found during EDA.
- Precision@K for the recommender is an approximation (keyword overlap between resume category and recommended job title), since no ground-truth resume-to-job pairing dataset exists.
- Clustering silhouette scores are low (~0.02–0.03), reflecting the fact that job postings don't separate into naturally distinct, well-separated clusters in TF-IDF space — cluster keywords are still directionally useful for the skill-gap module, but cluster boundaries should not be over-interpreted.
- The Fit Predictor's training labels are weakly/synthetically generated (top-similarity resume-job pairs are treated as "good fit," low-similarity pairs as "poor fit") since no real shortlisting-outcome dataset exists. Treat its scores as directional, not ground-truth shortlisting probability.

## Stretch Goals Not Implemented

- Salary Band Predictor
- LDA/NMF topic modeling
- Rule-based mentor chat
## Deployed Link of the project
- https://smarthireproject-9ikknpdtutd8ao3xfndcgj.streamlit.app
