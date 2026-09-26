"""Evaluate Category & Audience Classifiers on the canonical Real Test Set.

This script:
1. Loads the 24 real held-out test documents from data/processed/classification_dataset/classification_dataset.csv.
2. Evaluates the trained Category Classifier:
   - Accuracy, Precision (macro, weighted), Recall (macro, weighted), F1 (macro, weighted)
   - Detailed per-category classification report
   - Confusion matrix (numerical JSON & plotted PNG)
   - Detailed error analysis with document ID, title, true vs. pred, and confidence
3. Evaluates the trained Audience Classifier:
   - Micro F1, Macro F1, Weighted F1
   - Per-audience precision, recall, F1
   - Exact-match accuracy & Hamming loss
   - Multi-label error analysis with true vs. predicted label sets
4. Saves all structured metric reports, error CSVs, and visualization plots to reports/classification/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("evaluate_classifiers")

DATASET_PATH = Path("data/processed/classification_dataset/classification_dataset.csv")
CAT_MODEL_DIR = Path("models/category_classifier")
AUD_MODEL_DIR = Path("models/audience_classifier")
REPORT_DIR = Path("reports/classification")


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


def evaluate_category(test_df: pd.DataFrame) -> dict:
    LOG.info("Evaluating Category Classifier on %d test documents...", len(test_df))
    model = joblib.load(CAT_MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(CAT_MODEL_DIR / "vectorizer.joblib")

    X_test_vec = vectorizer.transform(test_df["cleaned_text"].values)
    y_true = test_df["category"].values

    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)
    classes = list(model.classes_)

    acc = accuracy_score(y_true, preds)
    macro_p = precision_score(y_true, preds, average="macro", zero_division=0)
    weighted_p = precision_score(y_true, preds, average="weighted", zero_division=0)
    macro_r = recall_score(y_true, preds, average="macro", zero_division=0)
    weighted_r = recall_score(y_true, preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)

    report_str = classification_report(y_true, preds, labels=classes, zero_division=0)
    report_dict = classification_report(y_true, preds, labels=classes, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_true, preds, labels=classes)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Category Confusion Matrix (Test Set - 24 Real Notices)", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold"
            )

    plt.ylabel("True Category", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = REPORT_DIR / "category_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Error analysis
    errors = []
    for idx, (doc_id, title, true_cat, pred_cat, prob_vec) in enumerate(
        zip(test_df["document_id"], test_df["title"], y_true, preds, probs)
    ):
        pred_idx = classes.index(pred_cat)
        conf = float(prob_vec[pred_idx])
        if true_cat != pred_cat:
            errors.append({
                "document_id": doc_id,
                "title": title,
                "true_category": true_cat,
                "predicted_category": pred_cat,
                "confidence": round(conf, 4),
                "true_category_confidence": round(float(prob_vec[classes.index(true_cat)]), 4),
            })

    error_df = pd.DataFrame(errors)
    error_df.to_csv(REPORT_DIR / "category_errors.csv", index=False)

    metrics = {
        "test_samples": len(test_df),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "weighted_precision": round(weighted_p, 4),
        "macro_recall": round(macro_r, 4),
        "weighted_recall": round(weighted_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_category": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1-score": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in classes if cls in report_dict
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist(),
        },
        "errors_count": len(errors),
    }

    with (REPORT_DIR / "category_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (REPORT_DIR / "category_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write("CATEGORY CLASSIFICATION REPORT (Test Set: 24 Real Documents)\n")
        f.write("=" * 65 + "\n\n")
        f.write(report_str)

    LOG.info("Category Test Results - Accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f, Errors: %d",
             acc, macro_f1, weighted_f1, len(errors))
    return metrics


def evaluate_audience(test_df: pd.DataFrame) -> dict:
    LOG.info("Evaluating Audience Classifier on %d test documents...", len(test_df))
    model = joblib.load(AUD_MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(AUD_MODEL_DIR / "vectorizer.joblib")
    mlb = joblib.load(AUD_MODEL_DIR / "mlb.joblib")

    X_test_vec = vectorizer.transform(test_df["cleaned_text"].values)
    raw_true_labels = parse_audience_labels(test_df["audience_labels"])
    y_true = mlb.transform(raw_true_labels)

    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)
    classes = list(mlb.classes_)

    micro_f1 = f1_score(y_true, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    exact_acc = accuracy_score(y_true, preds)
    h_loss = hamming_loss(y_true, preds)

    report_str = classification_report(y_true, preds, target_names=classes, zero_division=0)
    report_dict = classification_report(y_true, preds, target_names=classes, output_dict=True, zero_division=0)

    # Decode predictions
    pred_labels = mlb.inverse_transform(preds)

    errors = []
    for doc_id, title, true_set, pred_set, prob_row in zip(
        test_df["document_id"], test_df["title"], raw_true_labels, pred_labels, probs
    ):
        true_sorted = sorted(true_set)
        pred_sorted = sorted(list(pred_set))
        if true_sorted != pred_sorted:
            conf_dict = {cls: round(float(p), 4) for cls, p in zip(classes, prob_row)}
            errors.append({
                "document_id": doc_id,
                "title": title,
                "true_audience": json.dumps(true_sorted),
                "predicted_audience": json.dumps(pred_sorted),
                "confidences": json.dumps(conf_dict),
            })

    error_df = pd.DataFrame(errors)
    error_df.to_csv(REPORT_DIR / "audience_errors.csv", index=False)

    metrics = {
        "test_samples": len(test_df),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "exact_match_accuracy": round(exact_acc, 4),
        "hamming_loss": round(h_loss, 4),
        "per_audience": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1-score": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in classes if cls in report_dict
        },
        "errors_count": len(errors),
    }

    with (REPORT_DIR / "audience_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (REPORT_DIR / "audience_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write("AUDIENCE MULTI-LABEL CLASSIFICATION REPORT (Test Set: 24 Real Documents)\n")
        f.write("=" * 65 + "\n\n")
        f.write(report_str)

    LOG.info("Audience Test Results - Micro F1: %.4f, Macro F1: %.4f, Exact Match: %.4f, Hamming Loss: %.4f, Errors: %d",
             micro_f1, macro_f1, exact_acc, h_loss, len(errors))
    return metrics


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info("Loading canonical dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    test_df = df[df["split"] == "test"].copy()
    assert len(test_df) == 24, f"Expected 24 test documents, found {len(test_df)}"
    assert not test_df["is_synthetic"].any(), "Test set must not contain synthetic records!"

    evaluate_category(test_df)
    evaluate_audience(test_df)
    LOG.info("Evaluation complete. All reports and figures generated in %s", REPORT_DIR)


if __name__ == "__main__":
    main()
