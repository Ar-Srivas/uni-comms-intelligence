"""Train Category Classifier for University Communications Intelligence.

Task: Single-label classification into 6 categories:
['academic', 'admission', 'announcements', 'examination', 'general', 'hostel']

Model Architecture:
TF-IDF Vectorizer (sublinear_tf=False, ngram_range=(1, 2), min_df=2, max_df=0.9)
  -> Logistic Regression (class_weight='balanced', C=1.0, random_state=42)
  Compared with LinearSVC baseline on validation split.

Data Splitting & Leakage Prevention:
- Train ONLY on split == 'train' (108 documents: 78 real + 30 synthetic)
- Validation split (18 documents, 100% real) used for model selection
- Test split (24 documents, 100% real) remains untouched
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("train_category")

RANDOM_STATE = 42
DATASET_PATH = Path("data/processed/classification_dataset/classification_dataset.csv")
MODEL_DIR = Path("models/category_classifier")


def validate_splits(df: pd.DataFrame) -> None:
    """Pre-training integrity check ensuring no data leakage."""
    train_ids = set(df[df["split"] == "train"]["document_id"])
    val_ids = set(df[df["split"].isin(["val", "validation"])]["document_id"])
    test_ids = set(df[df["split"] == "test"]["document_id"])

    # 1. No overlap between splits
    assert not (train_ids & val_ids), "Leakage detected: train and validation share document IDs!"
    assert not (train_ids & test_ids), "Leakage detected: train and test share document IDs!"
    assert not (val_ids & test_ids), "Leakage detected: validation and test share document IDs!"

    # 2. Synthetic documents only in training
    non_train_synthetic = df[(df["split"] != "train") & (df["is_synthetic"] == True)]
    assert len(non_train_synthetic) == 0, "Leakage detected: synthetic records found outside train split!"

    LOG.info("Split integrity check passed: train=%d, val=%d, test=%d. Synthetic train-only: True",
             len(train_ids), len(val_ids), len(test_ids))


def train() -> None:
    LOG.info("Loading canonical dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    validate_splits(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"].isin(["val", "validation"])].copy()

    X_train = train_df["cleaned_text"].values
    y_train = train_df["category"].values

    X_val = val_df["cleaned_text"].values
    y_val = val_df["category"].values

    LOG.info("Fitting TF-IDF vectorizer on training text...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=False,
        min_df=2,
        max_df=0.9,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    LOG.info("Vocabulary size: %d features", len(vectorizer.vocabulary_))

    # Baseline 1: Logistic Regression
    LOG.info("Training Logistic Regression (C=1.0, balanced)...")
    clf_lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf_lr.fit(X_train_vec, y_train)
    val_preds_lr = clf_lr.predict(X_val_vec)
    lr_macro_f1 = f1_score(y_val, val_preds_lr, average="macro", zero_division=0)
    lr_acc = accuracy_score(y_val, val_preds_lr)

    LOG.info("Logistic Regression Validation - Accuracy: %.4f, Macro F1: %.4f", lr_acc, lr_macro_f1)

    # Baseline 2: LinearSVC comparison
    LOG.info("Training LinearSVC (C=1.0, balanced)...")
    clf_svc = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf_svc.fit(X_train_vec, y_train)
    val_preds_svc = clf_svc.predict(X_val_vec)
    svc_macro_f1 = f1_score(y_val, val_preds_svc, average="macro", zero_division=0)
    svc_acc = accuracy_score(y_val, val_preds_svc)

    LOG.info("LinearSVC Validation - Accuracy: %.4f, Macro F1: %.4f", svc_acc, svc_macro_f1)

    # Select Logistic Regression as primary model for probability calibration and interpretability
    selected_model = clf_lr
    LOG.info("Selected Logistic Regression as primary model (provides calibrated probabilities).")

    LOG.info("\nValidation Classification Report (Logistic Regression):\n%s",
             classification_report(y_val, val_preds_lr, zero_division=0))

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    vec_path = MODEL_DIR / "vectorizer.joblib"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(selected_model, model_path)
    joblib.dump(vectorizer, vec_path)

    metadata = {
        "model_type": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "ngram_range": [1, 2],
        "sublinear_tf": False,
        "classes": list(selected_model.classes_),
        "random_state": RANDOM_STATE,
        "validation_accuracy": round(lr_acc, 4),
        "validation_macro_f1": round(lr_macro_f1, 4),
        "linearsvc_validation_macro_f1": round(svc_macro_f1, 4),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOG.info("Saved category classifier model and metadata to %s", MODEL_DIR)


if __name__ == "__main__":
    train()
