"""Unified Inference Pipeline for University Communications Intelligence.

Predicts both:
1. Category (Single-label): academic, admission, announcements, examination, general, hostel
2. Target Audience (Multi-label): students, faculty, administrators

Supports:
- Python module usage: `from scripts.predict import NoticeClassifier, predict_notice`
- CLI raw text: `python scripts/predict.py --text "..."`
- CLI file input: `python scripts/predict.py --file notice.txt`
- CLI PDF input: `python scripts/predict.py --pdf notice.pdf`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.preprocess import clean_text, extract_pdf_text

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("predict")

CAT_MODEL_DIR = Path("models/category_classifier")
AUD_MODEL_DIR = Path("models/audience_classifier")


class NoticeClassifier:
    """Production inference engine for Document Category and Target Audience."""

    def __init__(
        self,
        cat_dir: Path = CAT_MODEL_DIR,
        aud_dir: Path = AUD_MODEL_DIR,
        audience_threshold: float = 0.5,
    ) -> None:
        self.cat_model = joblib.load(cat_dir / "model.joblib")
        self.cat_vec = joblib.load(cat_dir / "vectorizer.joblib")

        self.aud_model = joblib.load(aud_dir / "model.joblib")
        self.aud_vec = joblib.load(aud_dir / "vectorizer.joblib")
        self.mlb = joblib.load(aud_dir / "mlb.joblib")

        self.cat_classes: list[str] = list(self.cat_model.classes_)
        self.aud_classes: list[str] = list(self.mlb.classes_)
        self.audience_threshold = audience_threshold

    def predict(self, text: str) -> dict[str, Any]:
        """Classify a notice's text into category and target audiences."""
        cleaned, _ = clean_text(text)
        if not cleaned.strip():
            return {
                "error": "Input text contains no readable characters after normalization",
                "category": None,
                "audience": [],
            }

        # 1. Category Classification
        X_cat = self.cat_vec.transform([cleaned])
        cat_pred = str(self.cat_model.predict(X_cat)[0])
        cat_probs = self.cat_model.predict_proba(X_cat)[0]

        cat_conf = float(np.max(cat_probs))
        cat_prob_map = {cls: round(float(p), 4) for cls, p in zip(self.cat_classes, cat_probs)}

        # 2. Audience Multi-label Classification
        X_aud = self.aud_vec.transform([cleaned])
        aud_probs = self.aud_model.predict_proba(X_aud)[0]

        aud_conf_map = {cls: round(float(p), 4) for cls, p in zip(self.aud_classes, aud_probs)}

        # Multi-label thresholding
        selected_audiences = [
            cls for cls, p in zip(self.aud_classes, aud_probs)
            if p >= self.audience_threshold
        ]

        # Fallback to highest confidence label if none cross the threshold
        if not selected_audiences:
            best_idx = int(np.argmax(aud_probs))
            selected_audiences = [self.aud_classes[best_idx]]

        return {
            "category": cat_pred,
            "category_confidence": round(cat_conf, 4),
            "category_probabilities": cat_prob_map,
            "audience": sorted(selected_audiences),
            "audience_confidences": aud_conf_map,
        }


_DEFAULT_CLASSIFIER: Optional[NoticeClassifier] = None


def predict_notice(text: str) -> dict[str, Any]:
    """Convenience helper for one-off prediction."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = NoticeClassifier()
    return _DEFAULT_CLASSIFIER.predict(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Category and Target Audience for University Notices.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw text of the university notice")
    group.add_argument("--file", type=Path, help="Path to text or markdown file containing notice")
    group.add_argument("--pdf", type=Path, help="Path to university notice PDF file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for multi-label audience (default: 0.5)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classifier = NoticeClassifier(audience_threshold=args.threshold)

    if args.text:
        text = args.text
    elif args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.pdf:
        extracted = extract_pdf_text(args.pdf)
        text = extracted.get("raw_text", "")
    else:
        print("No input provided.", file=sys.stderr)
        return 1

    result = classifier.predict(text)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
