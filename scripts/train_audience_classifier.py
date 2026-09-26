"""Train Audience Classifier for University Communications Intelligence.

Task: Multi-label classification into 3 audience roles:
['administrators', 'faculty', 'students']

Model Architecture:
TF-IDF Vectorizer (sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
  -> OneVsRestClassifier(LogisticRegression(C=2.0, class_weight='balanced', random_state=42))
  Compared with OneVsRestClassifier(LinearSVC) baseline on validation split.

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("train_audience")

RANDOM_STATE = 42
DATASET_PATH = Path("data/processed/classification_dataset/classification_dataset.csv")
MODEL_DIR = Path("models/audience_classifier")


def validate_splits(df: pd.DataFrame) -> None:
    """Pre-training integrity check ensuring no data leakage."""
    train_ids = set(df[df["split"] == "train"]["document_id"])
    val_ids = set(df[df["split"].isin(["val", "validation"])]["document_id"])
    test_ids = set(df[df["split"] == "test"]["document_id"])

    assert not (train_ids & val_ids), "Leakage detected: train and validation share document IDs!"
    assert not (train_ids & test_ids), "Leakage detected: train and test share document IDs!"
    assert not (val_ids & test_ids), "Leakage detected: validation and test share document IDs!"

    non_train_synthetic = df[(df["split"] != "train") & (df["is_synthetic"] == True)]
    assert len(non_train_synthetic) == 0, "Leakage detected: synthetic records found outside train split!"


def parse_audience_labels(series: pd.Series) -> list[list[str]]:
    """Safely parse audience_labels from JSON strings or lists."""
    results = []
    for val in series:
        if isinstance(val, list):
            results.append(val)
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    results.append(parsed)
                else:
                    results.append([str(parsed)])
            except json.JSONDecodeError:
                cleaned = [x.strip(" '\"[]") for x in val.split(",") if x.strip(" '\"[]")]
                results.append(cleaned)
        else:
            results.append([])
    return results


def train() -> None:
    LOG.info("Loading canonical dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    validate_splits(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"].isin(["val", "validation"])].copy()

    X_train = train_df["cleaned_text"].values
    X_val = val_df["cleaned_text"].values

    raw_train_labels = parse_audience_labels(train_df["audience_labels"])
    raw_val_labels = parse_audience_labels(val_df["audience_labels"])

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(raw_train_labels)
    y_val = mlb.transform(raw_val_labels)

    LOG.info("Audience classes (%d): %s", len(mlb.classes_), list(mlb.classes_))

    LOG.info("Fitting TF-IDF vectorizer on training text...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.9,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    LOG.info("Vocabulary size: %d features", len(vectorizer.vocabulary_))

    # Baseline 1: OneVsRestClassifier(LogisticRegression)
    LOG.info("Training OneVsRest(LogisticRegression) (C=2.0, balanced)...")
    base_lr = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf_lr = OneVsRestClassifier(base_lr)
    clf_lr.fit(X_train_vec, y_train)

    val_preds_lr = clf_lr.predict(X_val_vec)
    macro_f1_lr = f1_score(y_val, val_preds_lr, average="macro", zero_division=0)
    micro_f1_lr = f1_score(y_val, val_preds_lr, average="micro", zero_division=0)
    exact_acc_lr = accuracy_score(y_val, val_preds_lr)
    h_loss_lr = hamming_loss(y_val, val_preds_lr)

    LOG.info("OneVsRest(LR) Validation - Macro F1: %.4f, Micro F1: %.4f, Exact Match: %.4f, Hamming: %.4f",
             macro_f1_lr, micro_f1_lr, exact_acc_lr, h_loss_lr)

    # Baseline 2: OneVsRestClassifier(LinearSVC)
    LOG.info("Training OneVsRest(LinearSVC) (C=1.0, balanced)...")
    base_svc = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf_svc = OneVsRestClassifier(base_svc)
    clf_svc.fit(X_train_vec, y_train)

    val_preds_svc = clf_svc.predict(X_val_vec)
    macro_f1_svc = f1_score(y_val, val_preds_svc, average="macro", zero_division=0)
    micro_f1_svc = f1_score(y_val, val_preds_svc, average="micro", zero_division=0)
    exact_acc_svc = accuracy_score(y_val, val_preds_svc)
    h_loss_svc = hamming_loss(y_val, val_preds_svc)

    LOG.info("OneVsRest(SVC) Validation - Macro F1: %.4f, Micro F1: %.4f, Exact Match: %.4f, Hamming: %.4f",
             macro_f1_svc, micro_f1_svc, exact_acc_svc, h_loss_svc)

    # Select Logistic Regression model for calibrated confidence scores and probabilities
    selected_model = clf_lr
    LOG.info("Selected OneVsRest(LogisticRegression) for calibrated multi-label probabilities.")

    LOG.info("\nValidation Classification Report (Audience - Logistic Regression):\n%s",
             classification_report(y_val, val_preds_lr, target_names=list(mlb.classes_), zero_division=0))

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    vec_path = MODEL_DIR / "vectorizer.joblib"
    mlb_path = MODEL_DIR / "mlb.joblib"
    meta_path = MODEL_DIR / "metadata.json"

    joblib.dump(selected_model, model_path)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(mlb, mlb_path)

    metadata = {
        "model_type": "OneVsRestClassifier(LogisticRegression)",
        "vectorizer": "TfidfVectorizer",
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "classes": list(mlb.classes_),
        "random_state": RANDOM_STATE,
        "validation_macro_f1": round(macro_f1_lr, 4),
        "validation_micro_f1": round(micro_f1_lr, 4),
        "validation_exact_match_accuracy": round(exact_acc_lr, 4),
        "validation_hamming_loss": round(h_loss_lr, 4),
        "svc_validation_macro_f1": round(macro_f1_svc, 4),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOG.info("Saved audience classifier model, vectorizer, mlb, and metadata to %s", MODEL_DIR)


if __name__ == "__main__":
    train()
