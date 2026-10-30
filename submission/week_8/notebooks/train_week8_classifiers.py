from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("train_week8_classifiers")

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = WORKSPACE_ROOT / "submission" / "week_8" / "data"
MODEL_DIR = WORKSPACE_ROOT / "submission" / "week_7" / "models"
CLASSIFIER_DATA_FILES = (
    "s1_classified_circulars_v2.jsonl",
    "faculty_tagged_new.jsonl",
)

CATEGORIES = ["academic", "admissions", "announcements", "examinations", "placements"]
AUDIENCES = ["admin", "faculty", "student"]
RANDOM_STATE = 42
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20
# These match the Week 7 API's runtime audience thresholds.
AUDIENCE_THRESHOLDS = {"admin": 0.5, "faculty": 0.25, "student": 0.5}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
    return records


def load_records() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for filename in CLASSIFIER_DATA_FILES:
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required Week 8 classifier data not found: {path}")

        for record in load_jsonl(path):
            required = ("doc_id", "primary_category", "target_audiences", "title_upweighted_text")
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"{path.name} record is missing fields {missing}: {record.get('doc_id')}")

            doc_id = str(record["doc_id"])
            category = record["primary_category"]
            audiences = record["target_audiences"]
            text = record["title_upweighted_text"]
            if category not in CATEGORIES:
                raise ValueError(f"Unsupported category {category!r} for {doc_id}")
            if not isinstance(audiences, list) or not audiences:
                raise ValueError(f"Expected non-empty target_audiences list for {doc_id}")
            unknown_audiences = set(audiences) - set(AUDIENCES)
            if unknown_audiences:
                raise ValueError(f"Unsupported audience labels {sorted(unknown_audiences)} for {doc_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing classifier text for {doc_id}")

            normalized = {
                "doc_id": doc_id,
                "category": category,
                "audiences": sorted(set(audiences)),
                "text": text.strip(),
            }
            existing = by_id.get(doc_id)
            if existing is not None and existing != normalized:
                raise ValueError(f"Conflicting duplicate document ID {doc_id} in Week 8 JSONL files")
            by_id[doc_id] = normalized

    # Keep identical notices in one example so copies cannot leak across splits.
    by_text: dict[str, dict[str, Any]] = {}
    for record in by_id.values():
        text_key = hashlib.sha256(
            re.sub(r"\s+", " ", record["text"]).strip().casefold().encode("utf-8")
        ).hexdigest()
        existing = by_text.get(text_key)
        if existing is not None:
            if (existing["category"], existing["audiences"]) != (
                record["category"],
                record["audiences"],
            ):
                raise ValueError(
                    f"Identical notice text has conflicting labels: "
                    f"{existing['doc_id']} and {record['doc_id']}"
                )
            continue
        by_text[text_key] = record

    records = list(by_text.values())
    if len(records) < 10:
        raise ValueError(f"Not enough unique labeled records to train: {len(records)}")
    LOG.info(
        "Loaded %d unique notices from %d Week 8 JSONL files",
        len(records),
        len(CLASSIFIER_DATA_FILES),
    )
    return records


def split_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_val, test = train_test_split(
        records,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=[record["category"] for record in records],
    )
    relative_validation_fraction = VALIDATION_FRACTION / (1.0 - TEST_FRACTION)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation_fraction,
        random_state=RANDOM_STATE,
        stratify=[record["category"] for record in train_val],
    )

    for name, split in (("train", train), ("validation", validation), ("test", test)):
        category_counts = {label: sum(row["category"] == label for row in split) for label in CATEGORIES}
        audience_counts = {
            label: sum(label in row["audiences"] for row in split) for label in AUDIENCES
        }
        LOG.info(
            "%s: %d rows | categories=%s | audiences=%s",
            name,
            len(split),
            category_counts,
            audience_counts,
        )
    return train, validation, test


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )


def train_category_model(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[TfidfVectorizer, LogisticRegression]:
    train_text = [record["text"] for record in train]
    validation_text = [record["text"] for record in validation]
    validation_labels = [record["category"] for record in validation]
    vectorizer = make_vectorizer()
    train_vectors = vectorizer.fit_transform(train_text)
    validation_vectors = vectorizer.transform(validation_text)

    candidates = {
        "standard": LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE
        ),
        "balanced": LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    scores: dict[str, tuple[float, float]] = {}
    for name, model in candidates.items():
        model.fit(train_vectors, [record["category"] for record in train])
        predictions = model.predict(validation_vectors)
        scores[name] = (
            f1_score(validation_labels, predictions, average="macro", zero_division=0),
            accuracy_score(validation_labels, predictions),
        )
        LOG.info(
            "Category %s validation: macro F1=%.4f accuracy=%.4f",
            name,
            *scores[name],
        )

    selected_name = max(scores, key=lambda name: scores[name])
    LOG.info("Selected %s category model using validation results", selected_name)

    final_records = train + validation
    final_vectorizer = make_vectorizer()
    final_train_vectors = final_vectorizer.fit_transform(
        [record["text"] for record in final_records]
    )
    final_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced" if selected_name == "balanced" else None,
        random_state=RANDOM_STATE,
    )
    final_model.fit(final_train_vectors, [record["category"] for record in final_records])

    test_vectors = final_vectorizer.transform([record["text"] for record in test])
    test_labels = [record["category"] for record in test]
    test_predictions = final_model.predict(test_vectors)
    LOG.info(
        "Category held-out test: macro F1=%.4f accuracy=%.4f\n%s",
        f1_score(test_labels, test_predictions, average="macro", zero_division=0),
        accuracy_score(test_labels, test_predictions),
        classification_report(test_labels, test_predictions, labels=CATEGORIES, zero_division=0),
    )
    return final_vectorizer, final_model


def audience_predictions(probabilities: np.ndarray) -> np.ndarray:
    predictions = np.zeros_like(probabilities, dtype=int)
    for index, label in enumerate(AUDIENCES):
        predictions[:, index] = (probabilities[:, index] >= AUDIENCE_THRESHOLDS[label]).astype(int)
    return predictions


def train_audience_model(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[TfidfVectorizer, OneVsRestClassifier, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer(classes=AUDIENCES)
    train_labels = mlb.fit_transform([record["audiences"] for record in train])
    validation_labels = mlb.transform([record["audiences"] for record in validation])
    train_vectors_for_selection = make_vectorizer()
    train_vectors = train_vectors_for_selection.fit_transform(
        [record["text"] for record in train]
    )
    validation_vectors = train_vectors_for_selection.transform(
        [record["text"] for record in validation]
    )

    candidates = {
        "standard": OneVsRestClassifier(
            LogisticRegression(C=2.0, max_iter=2000, random_state=RANDOM_STATE)
        ),
        "balanced": OneVsRestClassifier(
            LogisticRegression(
                C=2.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    }
    scores: dict[str, tuple[float, float]] = {}
    for name, model in candidates.items():
        model.fit(train_vectors, train_labels)
        predictions = audience_predictions(model.predict_proba(validation_vectors))
        scores[name] = (
            f1_score(validation_labels, predictions, average="macro", zero_division=0),
            f1_score(validation_labels, predictions, average="micro", zero_division=0),
        )
        LOG.info(
            "Audience %s validation (Week 7 thresholds): macro F1=%.4f micro F1=%.4f",
            name,
            *scores[name],
        )

    selected_name = max(scores, key=lambda name: scores[name])
    LOG.info("Selected %s audience model using validation results", selected_name)

    final_records = train + validation
    final_vectorizer = make_vectorizer()
    final_train_vectors = final_vectorizer.fit_transform(
        [record["text"] for record in final_records]
    )
    final_labels = mlb.transform([record["audiences"] for record in final_records])
    final_model = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced" if selected_name == "balanced" else None,
            random_state=RANDOM_STATE,
        )
    )
    final_model.fit(final_train_vectors, final_labels)

    test_vectors = final_vectorizer.transform([record["text"] for record in test])
    test_labels = mlb.transform([record["audiences"] for record in test])
    test_predictions = audience_predictions(final_model.predict_proba(test_vectors))
    LOG.info(
        "Audience held-out test: macro F1=%.4f micro F1=%.4f exact match=%.4f hamming loss=%.4f\n%s",
        f1_score(test_labels, test_predictions, average="macro", zero_division=0),
        f1_score(test_labels, test_predictions, average="micro", zero_division=0),
        accuracy_score(test_labels, test_predictions),
        hamming_loss(test_labels, test_predictions),
        classification_report(
            test_labels, test_predictions, target_names=AUDIENCES, zero_division=0
        ),
    )
    return final_vectorizer, final_model, mlb


def main() -> None:
    records = load_records()
    train, validation, test = split_records(records)
    cat_vectorizer, cat_model = train_category_model(train, validation, test)
    aud_vectorizer, aud_model, mlb = train_audience_model(train, validation, test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "cat_vec.joblib": cat_vectorizer,
        "cat_model.joblib": cat_model,
        "aud_vec.joblib": aud_vectorizer,
        "aud_model.joblib": aud_model,
        "mlb.joblib": mlb,
    }
    for filename, artifact in artifacts.items():
        joblib.dump(artifact, MODEL_DIR / filename)
        LOG.info("Saved %s", MODEL_DIR / filename)


if __name__ == "__main__":
    main()
