"""Train Week 6 Category & Audience Classifiers on College Email Data.

This script:
1. Loads the leakage-free train and validation splits from data/processed/classification_dataset/college_emails_dataset.csv.
2. Trains both Standard and Class-Weighted Logistic Regression for:
   - Category Classifier (Single-label across 5 categories)
   - Audience Classifier (Multi-label across student, faculty, admin)
3. Performs validation threshold tuning for minority audience labels (faculty, admin).
4. Selects the superior model on the validation split and saves all models, vectorizers, and metadata
   to dedicated Week 6 model directories (models/week6/), leaving shared/legacy models untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("train_week6")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE_ROOT / "data/processed/classification_dataset/college_emails_dataset.csv"
CAT_MODEL_DIR = WORKSPACE_ROOT / "models/week6/category_classifier"
AUD_MODEL_DIR = WORKSPACE_ROOT / "models/week6/audience_classifier"

CATEGORIES = ["academic", "admissions", "announcements", "examinations", "placements"]
AUDIENCES = ["admin", "faculty", "student"]
RANDOM_STATE = 42


def parse_audience_labels(series: pd.Series) -> list[list[str]]:
    results = []
    for val in series:
        if isinstance(val, list):
            results.append(val)
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                results.append(parsed if isinstance(parsed, list) else [str(parsed)])
            except json.JSONDecodeError:
                cleaned = [x.strip(" '\"[]") for x in val.split(",") if x.strip(" '\"[]")]
                results.append(cleaned)
        else:
            results.append([])
    return results


def train_category_classifiers(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    LOG.info("=== Training Category Classifiers (5 Classes) ===")
    CAT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train = train_df["text"].values
    y_train = train_df["category"].values
    X_val = val_df["text"].values
    y_val = val_df["category"].values

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    LOG.info("Category TF-IDF vocabulary size: %d features", len(vectorizer.vocabulary_))

    # Model 1: Standard Logistic Regression (class_weight=None)
    clf_std = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )
    clf_std.fit(X_train_vec, y_train)
    val_preds_std = clf_std.predict(X_val_vec)
    acc_std = accuracy_score(y_val, val_preds_std)
    macro_f1_std = f1_score(y_val, val_preds_std, average="macro", zero_division=0)
    weighted_f1_std = f1_score(y_val, val_preds_std, average="weighted", zero_division=0)

    LOG.info("Category Standard LR Validation - Accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
             acc_std, macro_f1_std, weighted_f1_std)

    # Model 2: Class-Weighted Logistic Regression (class_weight='balanced')
    clf_bal = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf_bal.fit(X_train_vec, y_train)
    val_preds_bal = clf_bal.predict(X_val_vec)
    acc_bal = accuracy_score(y_val, val_preds_bal)
    macro_f1_bal = f1_score(y_val, val_preds_bal, average="macro", zero_division=0)
    weighted_f1_bal = f1_score(y_val, val_preds_bal, average="weighted", zero_division=0)

    LOG.info("Category Balanced LR Validation - Accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
             acc_bal, macro_f1_bal, weighted_f1_bal)

    # Save both models and vectorizer
    joblib.dump(clf_std, CAT_MODEL_DIR / "model_standard.joblib")
    joblib.dump(clf_bal, CAT_MODEL_DIR / "model_balanced.joblib")
    joblib.dump(vectorizer, CAT_MODEL_DIR / "vectorizer.joblib")

    # Select best model based on validation macro F1 (tie-breaker: accuracy)
    best_name = "balanced" if macro_f1_bal >= macro_f1_std else "standard"
    best_clf = clf_bal if best_name == "balanced" else clf_std
    joblib.dump(best_clf, CAT_MODEL_DIR / "model.joblib")
    LOG.info("Selected '%s' category classifier as primary model.", best_name)

    cat_metadata = {
        "model_type": f"LogisticRegression(class_weight='{clf_bal.class_weight if best_name == 'balanced' else None}')",
        "selected_model": best_name,
        "classes": CATEGORIES,
        "random_state": RANDOM_STATE,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "max_df": 0.9,
            "vocab_size": len(vectorizer.vocabulary_),
        },
        "validation_comparison": {
            "standard": {
                "accuracy": round(float(acc_std), 4),
                "macro_f1": round(float(macro_f1_std), 4),
                "weighted_f1": round(float(weighted_f1_std), 4),
            },
            "balanced": {
                "accuracy": round(float(acc_bal), 4),
                "macro_f1": round(float(macro_f1_bal), 4),
                "weighted_f1": round(float(weighted_f1_bal), 4),
            },
        },
    }
    with (CAT_MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(cat_metadata, f, indent=2)

    return cat_metadata


def tune_audience_thresholds(clf, X_val_vec, y_val, classes: list[str]) -> dict[str, float]:
    """Find per-class decision thresholds that maximize F1 on the validation set."""
    val_probs = clf.predict_proba(X_val_vec)
    best_thresholds = {cls: 0.5 for cls in classes}

    for i, cls in enumerate(classes):
        best_f1 = -1.0
        best_th = 0.5
        for th in np.arange(0.20, 0.65, 0.05):
            preds = (val_probs[:, i] >= th).astype(int)
            f1 = f1_score(y_val[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = round(float(th), 2)
        best_thresholds[cls] = best_th

    LOG.info("Validation tuned audience thresholds: %s", best_thresholds)
    return best_thresholds


def train_audience_classifiers(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    LOG.info("=== Training Audience Classifiers (Multi-Label) ===")
    AUD_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train = train_df["text"].values
    X_val = val_df["text"].values

    raw_y_train = parse_audience_labels(train_df["audience_labels"])
    raw_y_val = parse_audience_labels(val_df["audience_labels"])

    mlb = MultiLabelBinarizer(classes=AUDIENCES)
    y_train = mlb.fit_transform(raw_y_train)
    y_val = mlb.transform(raw_y_val)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    LOG.info("Audience TF-IDF vocabulary size: %d features", len(vectorizer.vocabulary_))

    # Model 1: Standard OneVsRest Logistic Regression
    clf_std = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight=None,
            random_state=RANDOM_STATE,
        )
    )
    clf_std.fit(X_train_vec, y_train)
    val_preds_std = clf_std.predict(X_val_vec)

    # Standard default 0.5 threshold evaluation
    acc_std = accuracy_score(y_val, val_preds_std)
    hamming_std = hamming_loss(y_val, val_preds_std)
    macro_f1_std = f1_score(y_val, val_preds_std, average="macro", zero_division=0)
    micro_f1_std = f1_score(y_val, val_preds_std, average="micro", zero_division=0)

    LOG.info("Audience Standard LR Validation - ExactMatch: %.4f, HammingLoss: %.4f, Macro F1: %.4f, Micro F1: %.4f",
             acc_std, hamming_std, macro_f1_std, micro_f1_std)

    # Model 2: Balanced OneVsRest Logistic Regression
    clf_bal = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )
    clf_bal.fit(X_train_vec, y_train)

    # Tune thresholds on validation split
    best_thresholds = tune_audience_thresholds(clf_bal, X_val_vec, y_val, AUDIENCES)

    # Predict with tuned thresholds
    bal_probs = clf_bal.predict_proba(X_val_vec)
    val_preds_bal = np.zeros_like(y_val)
    for i, cls in enumerate(AUDIENCES):
        val_preds_bal[:, i] = (bal_probs[:, i] >= best_thresholds[cls]).astype(int)

    # Ensure fallback to 'student' if no label crosses threshold
    student_idx = AUDIENCES.index("student")
    for row_idx in range(len(val_preds_bal)):
        if val_preds_bal[row_idx].sum() == 0:
            val_preds_bal[row_idx, student_idx] = 1

    acc_bal = accuracy_score(y_val, val_preds_bal)
    hamming_bal = hamming_loss(y_val, val_preds_bal)
    macro_f1_bal = f1_score(y_val, val_preds_bal, average="macro", zero_division=0)
    micro_f1_bal = f1_score(y_val, val_preds_bal, average="micro", zero_division=0)

    LOG.info("Audience Balanced LR Validation - ExactMatch: %.4f, HammingLoss: %.4f, Macro F1: %.4f, Micro F1: %.4f",
             acc_bal, hamming_bal, macro_f1_bal, micro_f1_bal)

    # Save artifacts
    joblib.dump(clf_std, AUD_MODEL_DIR / "model_standard.joblib")
    joblib.dump(clf_bal, AUD_MODEL_DIR / "model_balanced.joblib")
    joblib.dump(vectorizer, AUD_MODEL_DIR / "vectorizer.joblib")
    joblib.dump(mlb, AUD_MODEL_DIR / "mlb.joblib")

    with (AUD_MODEL_DIR / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2)

    # Primary model selection: balanced with tuned thresholds
    best_name = "balanced_tuned" if macro_f1_bal >= macro_f1_std else "standard"
    best_clf = clf_bal if best_name == "balanced_tuned" else clf_std
    joblib.dump(best_clf, AUD_MODEL_DIR / "model.joblib")
    LOG.info("Selected '%s' audience classifier as primary model.", best_name)

    aud_metadata = {
        "model_type": f"OneVsRestClassifier(LogisticRegression(class_weight='{clf_bal.estimator.class_weight if best_name == 'balanced_tuned' else None}'))",
        "selected_model": best_name,
        "classes": AUDIENCES,
        "tuned_thresholds": best_thresholds,
        "random_state": RANDOM_STATE,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "max_df": 0.9,
            "vocab_size": len(vectorizer.vocabulary_),
        },
        "validation_comparison": {
            "standard_default_threshold": {
                "exact_match_accuracy": round(float(acc_std), 4),
                "hamming_loss": round(float(hamming_std), 4),
                "macro_f1": round(float(macro_f1_std), 4),
                "micro_f1": round(float(micro_f1_std), 4),
            },
            "balanced_tuned_threshold": {
                "exact_match_accuracy": round(float(acc_bal), 4),
                "hamming_loss": round(float(hamming_bal), 4),
                "macro_f1": round(float(macro_f1_bal), 4),
                "micro_f1": round(float(micro_f1_bal), 4),
            },
        },
    }
    with (AUD_MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(aud_metadata, f, indent=2)

    return aud_metadata


def main() -> None:
    LOG.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    LOG.info("Train set: %d rows | Validation set: %d rows", len(train_df), len(val_df))

    cat_meta = train_category_classifiers(train_df, val_df)
    aud_meta = train_audience_classifiers(train_df, val_df)

    LOG.info("Training complete. Artifacts saved to:")
    LOG.info("  Category: %s", CAT_MODEL_DIR)
    LOG.info("  Audience: %s", AUD_MODEL_DIR)


if __name__ == "__main__":
    main()
