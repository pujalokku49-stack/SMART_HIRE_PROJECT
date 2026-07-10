from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
import numpy as np


def evaluate_classifier(y_true, y_pred, y_proba=None, labels=None):
    """
    Compute standard classification metrics.
    y_proba: predicted probabilities (n_samples, n_classes) - needed for ROC-AUC.
    labels: ordered list of class labels matching y_proba's columns.
    """
    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    results["precision_weighted"] = precision
    results["recall_weighted"] = recall
    results["f1_weighted"] = f1

    results["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0
    )
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)

    if y_proba is not None:
        try:
            results["roc_auc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=labels
            )
        except ValueError as e:
            results["roc_auc_macro"] = None
            results["roc_auc_error"] = str(e)

    return results


def evaluate_clustering(vectors, labels, sample_size=2000, random_state=42):
    """Compute clustering quality metrics."""
    score = silhouette_score(
        vectors, labels, sample_size=sample_size, random_state=random_state
    )
    return {"silhouette_score": score}


def precision_at_k(recommend_fn, resumes_df, k=10, n_samples=50, random_state=42):
    """
    Proxy Precision@K: checks whether recommended job titles share a
    category-related word with the resume's known category.
    NOTE: this is an approximation, not ground-truth precision, since no
    dataset exists pairing resumes to a single 'correct' job.
    """
    sample = resumes_df.sample(n=n_samples, random_state=random_state)
    hits, total = 0, 0

    for _, row in sample.iterrows():
        category_words = row["Category"].lower().replace("-", " ").split()
        recs = recommend_fn(row["Resume"], top_n=k)
        for title in recs["title"]:
            total += 1
            if any(word in str(title).lower() for word in category_words):
                hits += 1

    return hits / total if total > 0 else 0