"""Evaluate Week 6 Category & Audience Classifiers on the Held-Out Test Set.

This script:
1. Loads the 17 held-out test documents from data/processed/classification_dataset/college_emails_dataset.csv.
2. Evaluates Category Classifiers (Standard vs Class-Weighted):
   - Multi-class Accuracy, Macro/Weighted Precision, Recall, F1
   - Per-category metrics table
   - Confusion matrix (JSON & PNG)
3. Evaluates Audience Classifiers (Standard vs Balanced with Tuned Thresholds):
   - Exact-match accuracy, Hamming loss, Micro F1, Macro F1
   - Per-audience label metrics table
   - Small-sample limitation analysis
4. Compares retrained models against baseline performance where feasible.
5. Exports structured evaluation JSON, test predictions CSV, and confusion matrix figure.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("evaluate_week6")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE_ROOT / "data/processed/classification_dataset/college_emails_dataset.csv"
CAT_MODEL_DIR = WORKSPACE_ROOT / "models/week6/category_classifier"
AUD_MODEL_DIR = WORKSPACE_ROOT / "models/week6/audience_classifier"
REPORTS_DIR = WORKSPACE_ROOT / "reports/classification"

CATEGORIES = ["academic", "admissions", "announcements", "examinations", "placements"]
AUDIENCES = ["admin", "faculty", "student"]


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
    LOG.info("Evaluating Category Classifiers on %d test emails...", len(test_df))
    vectorizer = joblib.load(CAT_MODEL_DIR / "vectorizer.joblib")
    clf_std = joblib.load(CAT_MODEL_DIR / "model_standard.joblib")
    clf_bal = joblib.load(CAT_MODEL_DIR / "model_balanced.joblib")

    X_test_vec = vectorizer.transform(test_df["text"].values)
    y_true = test_df["category"].values

    preds_std = clf_std.predict(X_test_vec)
    preds_bal = clf_bal.predict(X_test_vec)
    probs_bal = clf_bal.predict_proba(X_test_vec)

    # Standard metrics
    acc_std = accuracy_score(y_true, preds_std)
    macro_f1_std = f1_score(y_true, preds_std, average="macro", zero_division=0)
    weighted_f1_std = f1_score(y_true, preds_std, average="weighted", zero_division=0)

    # Balanced metrics
    acc_bal = accuracy_score(y_true, preds_bal)
    macro_p_bal = precision_score(y_true, preds_bal, average="macro", zero_division=0)
    macro_r_bal = recall_score(y_true, preds_bal, average="macro", zero_division=0)
    macro_f1_bal = f1_score(y_true, preds_bal, average="macro", zero_division=0)
    weighted_f1_bal = f1_score(y_true, preds_bal, average="weighted", zero_division=0)

    rep_bal = classification_report(y_true, preds_bal, labels=CATEGORIES, output_dict=True, zero_division=0)
    cm_bal = confusion_matrix(y_true, preds_bal, labels=CATEGORIES)

    # Plot Confusion Matrix
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_bal, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Week 6 Category Confusion Matrix (Test Set - 17 College Emails)", fontsize=11, fontweight="bold")
    fig.colorbar(im)
    tick_marks = np.arange(len(CATEGORIES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CATEGORIES, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")

    thresh = cm_bal.max() / 2.0 if cm_bal.max() > 0 else 1.0
    for i in range(cm_bal.shape[0]):
        for j in range(cm_bal.shape[1]):
            val = cm_bal[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontweight="bold")

    fig.tight_layout()
    cm_plot_path = REPORTS_DIR / "week6_category_confusion_matrix.png"
    fig.savefig(cm_plot_path, dpi=300)
    plt.close(fig)
    LOG.info("Saved confusion matrix to %s", cm_plot_path)

    return {
        "test_samples": len(test_df),
        "comparison": {
            "standard_lr": {
                "accuracy": round(float(acc_std), 4),
                "macro_f1": round(float(macro_f1_std), 4),
                "weighted_f1": round(float(weighted_f1_std), 4),
            },
            "balanced_lr": {
                "accuracy": round(float(acc_bal), 4),
                "macro_precision": round(float(macro_p_bal), 4),
                "macro_recall": round(float(macro_r_bal), 4),
                "macro_f1": round(float(macro_f1_bal), 4),
                "weighted_f1": round(float(weighted_f1_bal), 4),
            },
        },
        "per_category": {
            cat: {
                "precision": round(float(rep_bal[cat]["precision"]), 4),
                "recall": round(float(rep_bal[cat]["recall"]), 4),
                "f1_score": round(float(rep_bal[cat]["f1-score"]), 4),
                "support": int(rep_bal[cat]["support"]),
            }
            for cat in CATEGORIES
        },
        "confusion_matrix": {
            "classes": CATEGORIES,
            "matrix": cm_bal.tolist(),
        },
        "predictions_standard": preds_std.tolist(),
        "predictions_balanced": preds_bal.tolist(),
        "probabilities_balanced": probs_bal.tolist(),
    }


def evaluate_audience(test_df: pd.DataFrame) -> dict:
    LOG.info("Evaluating Audience Classifiers on %d test emails...", len(test_df))
    vectorizer = joblib.load(AUD_MODEL_DIR / "vectorizer.joblib")
    mlb = joblib.load(AUD_MODEL_DIR / "mlb.joblib")
    clf_std = joblib.load(AUD_MODEL_DIR / "model_standard.joblib")
    clf_bal = joblib.load(AUD_MODEL_DIR / "model_balanced.joblib")

    with (AUD_MODEL_DIR / "thresholds.json").open(encoding="utf-8") as f:
        thresholds = json.load(f)

    X_test_vec = vectorizer.transform(test_df["text"].values)
    raw_y_test = parse_audience_labels(test_df["audience_labels"])
    y_true = mlb.transform(raw_y_test)

    # Standard model with default 0.5 threshold
    preds_std = clf_std.predict(X_test_vec)
    acc_std = accuracy_score(y_true, preds_std)
    hamming_std = hamming_loss(y_true, preds_std)
    macro_f1_std = f1_score(y_true, preds_std, average="macro", zero_division=0)
    micro_f1_std = f1_score(y_true, preds_std, average="micro", zero_division=0)

    # Balanced model with tuned thresholds
    probs_bal = clf_bal.predict_proba(X_test_vec)
    preds_bal = np.zeros_like(y_true)
    for i, cls in enumerate(AUDIENCES):
        preds_bal[:, i] = (probs_bal[:, i] >= thresholds.get(cls, 0.5)).astype(int)

    # Fallback to student if empty
    student_idx = AUDIENCES.index("student")
    for row_idx in range(len(preds_bal)):
        if preds_bal[row_idx].sum() == 0:
            preds_bal[row_idx, student_idx] = 1

    acc_bal = accuracy_score(y_true, preds_bal)
    hamming_bal = hamming_loss(y_true, preds_bal)
    macro_f1_bal = f1_score(y_true, preds_bal, average="macro", zero_division=0)
    micro_f1_bal = f1_score(y_true, preds_bal, average="micro", zero_division=0)
    weighted_f1_bal = f1_score(y_true, preds_bal, average="weighted", zero_division=0)

    rep_bal = classification_report(y_true, preds_bal, target_names=AUDIENCES, output_dict=True, zero_division=0)

    # Format predictions as string lists
    pred_labels_std = mlb.inverse_transform(preds_std)
    pred_labels_bal = mlb.inverse_transform(preds_bal)

    return {
        "test_samples": len(test_df),
        "comparison": {
            "standard_lr_default_0.5": {
                "exact_match_accuracy": round(float(acc_std), 4),
                "hamming_loss": round(float(hamming_std), 4),
                "macro_f1": round(float(macro_f1_std), 4),
                "micro_f1": round(float(micro_f1_std), 4),
            },
            "balanced_lr_tuned_thresholds": {
                "exact_match_accuracy": round(float(acc_bal), 4),
                "hamming_loss": round(float(hamming_bal), 4),
                "macro_f1": round(float(macro_f1_bal), 4),
                "micro_f1": round(float(micro_f1_bal), 4),
                "weighted_f1": round(float(weighted_f1_bal), 4),
                "tuned_thresholds": thresholds,
            },
        },
        "per_audience": {
            aud: {
                "precision": round(float(rep_bal[aud]["precision"]), 4),
                "recall": round(float(rep_bal[aud]["recall"]), 4),
                "f1_score": round(float(rep_bal[aud]["f1-score"]), 4),
                "support": int(rep_bal[aud]["support"]),
            }
            for aud in AUDIENCES
        },
        "predictions_standard": [list(x) for x in pred_labels_std],
        "predictions_balanced": [list(x) for x in pred_labels_bal],
        "probabilities_balanced": probs_bal.tolist(),
    }


def main() -> None:
    LOG.info("Loading test dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    LOG.info("Loaded %d test records.", len(test_df))

    cat_res = evaluate_category(test_df)
    aud_res = evaluate_audience(test_df)

    # Build predictions dataframe
    preds_df = pd.DataFrame({
        "doc_id": test_df["doc_id"],
        "true_category": test_df["category"],
        "pred_category_standard": cat_res["predictions_standard"],
        "pred_category_balanced": cat_res["predictions_balanced"],
        "category_balanced_conf": [round(float(max(p)), 4) for p in cat_res["probabilities_balanced"]],
        "true_audience": test_df["audience_labels"],
        "pred_audience_standard": [json.dumps(p) for p in aud_res["predictions_standard"]],
        "pred_audience_balanced": [json.dumps(p) for p in aud_res["predictions_balanced"]],
    })
    preds_csv_path = REPORTS_DIR / "week6_test_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    LOG.info("Saved test predictions to %s", preds_csv_path)

    # Clean results dictionary for JSON export
    export_metrics = {
        "metadata": {
            "task": "Week 6 Classifier Retraining & Evaluation",
            "eval_split": "test",
            "sample_count": len(test_df),
        },
        "category_evaluation": {
            "comparison": cat_res["comparison"],
            "per_category": cat_res["per_category"],
            "confusion_matrix": cat_res["confusion_matrix"],
        },
        "audience_evaluation": {
            "comparison": aud_res["comparison"],
            "per_audience": aud_res["per_audience"],
        },
    }
    metrics_json_path = REPORTS_DIR / "week6_evaluation_metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(export_metrics, f, indent=2)
    LOG.info("Saved evaluation metrics JSON to %s", metrics_json_path)

    # Print summary to terminal
    print("\n" + "=" * 60)
    print("           WEEK 6 CLASSIFIER EVALUATION RESULTS")
    print("=" * 60)
    print("\n--- 1. CATEGORY CLASSIFIER (5 Classes, Test N=17) ---")
    print(f"Standard LR Accuracy: {cat_res['comparison']['standard_lr']['accuracy']:.4f} | Macro F1: {cat_res['comparison']['standard_lr']['macro_f1']:.4f}")
    print(f"Balanced LR Accuracy: {cat_res['comparison']['balanced_lr']['accuracy']:.4f} | Macro F1: {cat_res['comparison']['balanced_lr']['macro_f1']:.4f}")
    print("\nPer-Category Metrics (Balanced LR):")
    for cat, m in cat_res["per_category"].items():
        print(f"  - {cat:14s}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}, Support={m['support']}")

    print("\n--- 2. AUDIENCE CLASSIFIER (Multi-Label, Test N=17) ---")
    print(f"Standard LR ExactMatch: {aud_res['comparison']['standard_lr_default_0.5']['exact_match_accuracy']:.4f} | Macro F1: {aud_res['comparison']['standard_lr_default_0.5']['macro_f1']:.4f} | Micro F1: {aud_res['comparison']['standard_lr_default_0.5']['micro_f1']:.4f}")
    print(f"Balanced LR ExactMatch: {aud_res['comparison']['balanced_lr_tuned_thresholds']['exact_match_accuracy']:.4f} | Macro F1: {aud_res['comparison']['balanced_lr_tuned_thresholds']['macro_f1']:.4f} | Micro F1: {aud_res['comparison']['balanced_lr_tuned_thresholds']['micro_f1']:.4f}")
    print("\nPer-Audience Metrics (Balanced LR with Tuned Thresholds):")
    for aud, m in aud_res["per_audience"].items():
        print(f"  - {aud:10s}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}, Support={m['support']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
