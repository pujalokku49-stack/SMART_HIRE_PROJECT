import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import streamlit as st
from parsing.resume_parser import extract_resume_text
from models.classifier import ResumeClassifier
from models.recommender import JobRecommender
from models.fit_predictor import FitPredictor


st.set_page_config(page_title="SmartHire", page_icon="🎯", layout="wide")

# Load models once and cache them across reruns (Streamlit reruns the whole script on every interaction)
@st.cache_resource
def load_classifier():
    return ResumeClassifier()

@st.cache_resource
def load_recommender():
    return JobRecommender()

@st.cache_resource
def load_fit_predictor():
    return FitPredictor()

classifier = load_classifier()
recommender = load_recommender()
fit_predictor = load_fit_predictor()

st.title("🎯 SmartHire — Resume-to-Job Matching & Career Guidance")
st.write("Upload your resume to get matched jobs, a predicted career category, a fit score, and a skill-gap report.")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    with st.spinner("Reading your resume..."):
        resume_text = extract_resume_text(uploaded_file)

    if not resume_text.strip():
        st.error("Couldn't extract any text from this file. Try a different format or a text-based PDF (not a scanned image).")
    else:
        st.success("Resume loaded successfully!")

        # --- Section 1: Predicted Category ---
        st.header("📌 Predicted Career Category")
        category, confidence = classifier.predict(resume_text)
        st.metric("Predicted Category", category)

        # --- Section 2: Recommended Jobs + Fit Score ---
        st.header("💼 Top Matching Jobs")
        top_n = st.slider("Number of jobs to show", 5, 20, 10)

        with st.spinner("Scoring job fit..."):
            recommendations = recommender.recommend(resume_text, top_n=top_n, include_fit_columns=True)
            recommendations["fit_score"] = recommendations.apply(
                lambda row: fit_predictor.predict_fit(resume_text, row), axis=1
            )

        display_df = recommendations[["title", "company", "location", "source", "match_score", "fit_score"]].copy()
        display_df["match_score"] = display_df["match_score"].round(3)
        display_df["fit_score"] = display_df["fit_score"].round(3)
        st.dataframe(display_df, use_container_width=True)
        st.caption(
            "Match score = TF-IDF cosine similarity between resume and job description. "
            "Fit score = predicted probability of a good fit from the shortlisting model "
            "(trained on weak/synthetic labels — treat as directional, not a guarantee)."
        )

        # --- Section 3: Skill Gap Report ---
        st.header("📈 Skill Gap Report")
        report = recommender.skill_gap_report(resume_text)

        st.write("**Based on jobs like:**", ", ".join(report["based_on_jobs"][:3]))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("✅ Skills you already show")
            for skill in report["matched_skills"]:
                st.write(f"- {skill}")
        with col4:
            st.subheader("🎯 Skills to consider adding")
            for skill in report["missing_skills"]:
                st.write(f"- {skill}")
else:
    st.info("👆 Upload a resume (PDF, DOCX, or TXT) to get started.")