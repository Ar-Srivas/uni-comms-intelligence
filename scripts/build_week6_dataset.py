"""Build Week 6 College Emails Classification Dataset.

This script parses and validates the 114 college email notices from the new college email dataset,
maps them to the five intended category labels (academic, placements, admissions, announcements, examinations),
preserves natural multi-label audience distribution (student, faculty, admin),
and generates a strictly leakage-free train/val/test split.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("build_week6_dataset")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SOURCE_JSONL = WORKSPACE_ROOT / "submission/week_6/data/s1_classified_circulars.jsonl"
OUTPUT_DIR = WORKSPACE_ROOT / "data/processed/classification_dataset"
OUTPUT_CSV = OUTPUT_DIR / "college_emails_dataset.csv"
OUTPUT_JSONL = OUTPUT_DIR / "college_emails_dataset.jsonl"
REPORT_JSON = OUTPUT_DIR / "college_emails_dataset_report.json"

CATEGORIES = ["academic", "admissions", "announcements", "examinations", "placements"]
AUDIENCES = ["admin", "faculty", "student"]

# Duplicate & near-duplicate clusters to enforce strictly into train split (prevent leakage)
FORCE_TRAIN_IDS = {
    "adm-018-symbiosis-institute-of-technology",
    "adm-019-symbiosis-institute-of-technology",
    "ann-013-portal-announcement",
    "ann-020-portal-announcement",
    "plac-002-amendment-to-placement-policy-placement-season-202",
    "plac-003-amendment-to-internship-placement-policy-placement",
    "plac-004-amendment-to-placement-policy-placement-season-202",
    "ann-001-mpstme-odd-semester-final-exam-timetable-a-y-2026-",
    "ann-004-mpstme-odd-semester-final-exam-timetable-a-y-2026-",
    "exam-002-odd-sem-timetable-computer-based-subjects-all-prog",
    "exam-003-odd-sem-timetable-computer-based-subjects-all-prog",
    "plac-013-letter-of-recommendation-for-academic-internship-b",
}

# 17 Val IDs: 5 examinations, 5 placements, 3 admissions, 2 announcements, 2 academic
VAL_IDS = {
    "acad-017-class-rescheduling-for-data-structures-lab",         # academic (faculty)
    "acad-002-user-manual-photocopy-url-on-nmims-grievance-redre", # academic (student)
    "adm-001-dil-informative-session-study-opportunities-at-bin", # admissions
    "adm-002-uc-berkeley-summer-sessions-applications-open",      # admissions
    "adm-004-b-tech-mba-tech-admission-confirmation",             # admissions
    "ann-012-sattva-artist-reveal",                                # announcements (admin)
    "ann-007-library-newsletter-nmims-knowledge-trendsetters-vo", # announcements (student)
    "exam-004-re-test-for-m2-for-oe5-journalism-and-mass-media-f",# examinations (faculty)
    "acad-009-mid-semester-examination-timetable-semester-iii",   # examinations (admin)
    "exam-001-odd-sem-timetable-computer-based-subjects-all-prog",# examinations
    "ann-003-revised-mpstme-computer-based-re-exam-time-table-e",  # examinations
    "exam-007-class-test-2-m2-schedule-of-b-tech-mba-tech-ds-sem",# examinations
    "plac-001-technical-training-for-capgemini-nmims-mumbai-2027",# placements
    "plac-005-company-specific-training-sessions-for-kpmg",       # placements
    "plac-006-hirepro-technical-assessment-placement-training-pr",# placements
    "plac-007-online-aptitude-training-session-13-and-14-june-20",# placements
    "plac-008-cocubes-gamified-technical-assessment-placement-tr",# placements
}

# 17 Test IDs: 5 examinations, 5 placements, 3 admissions, 2 announcements, 2 academic
TEST_IDS = {
    "acad-019-industrial-visit-to-manufacturing-plant-pune-conse", # academic (faculty)
    "acad-005-students-registrations-for-academic-bank-of-credit", # academic (student)
    "adm-003-dil-informative-session-with-santa-clara-universit", # admissions (admin)
    "adm-005-admission-counselling-invitation-letter-for-b-tech", # admissions
    "adm-006-informative-session-the-university-of-new-south-wa", # admissions
    "ann-014-access-to-pai-s-friends-library",                     # announcements (faculty)
    "ann-006-the-monthly-edge-ai-newsletter",                     # announcements
    "exam-027-student-instructions-for-odd-semester-november-dec",# examinations (faculty, admin)
    "exam-009-13th-july-2026-seating-plan",                       # examinations
    "exam-010-seating-plan-14-july-2026",                         # examinations
    "exam-011-seating-plan-for-even-semester-re-exam-june-2026",  # examinations
    "exam-012-seating-plan-for-even-semester-re-exam-june-2026",  # examinations
    "plac-009-post-assessment-placement-training-program-batch-2",# placements
    "plac-010-crash-course-placement-training-6-to-14-july-2026", # placements
    "plac-011-lyondellbasell-hackathon-information-session",       # placements
    "plac-012-reminder-apply-for-lyondellbasell-hackathon",       # placements
    "plac-014-tcs-codevita-season-14-registrations-open",          # placements
}


def build_dataset() -> pd.DataFrame:
    LOG.info("Loading records from %s", SOURCE_JSONL)
    with SOURCE_JSONL.open(encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f]

    LOG.info("Loaded %d records.", len(raw_records))
    all_doc_ids = {r["doc_id"] for r in raw_records}

    assert len(raw_records) == 114, f"Expected 114 records, got {len(raw_records)}"
    assert VAL_IDS.issubset(all_doc_ids), f"Unknown val doc IDs: {VAL_IDS - all_doc_ids}"
    assert TEST_IDS.issubset(all_doc_ids), f"Unknown test doc IDs: {TEST_IDS - all_doc_ids}"
    assert len(VAL_IDS) == 17, f"Expected 17 val docs, got {len(VAL_IDS)}"
    assert len(TEST_IDS) == 17, f"Expected 17 test docs, got {len(TEST_IDS)}"
    assert len(VAL_IDS & TEST_IDS) == 0, "Overlap between validation and test!"
    assert len(FORCE_TRAIN_IDS & (VAL_IDS | TEST_IDS)) == 0, "Duplicate cluster leaked into val/test!"

    rows = []
    for r in raw_records:
        doc_id = r["doc_id"]
        category = r["primary_category"]
        audiences = sorted(r["target_audiences"])

        assert category in CATEGORIES, f"Invalid category {category} in {doc_id}"
        for a in audiences:
            assert a in AUDIENCES, f"Invalid audience {a} in {doc_id}"

        if doc_id in VAL_IDS:
            split = "val"
        elif doc_id in TEST_IDS:
            split = "test"
        else:
            split = "train"

        rows.append({
            "doc_id": doc_id,
            "category": category,
            "audience_labels": json.dumps(audiences),
            "text": r["title_upweighted_text"],
            "split": split,
            "num_characters": len(r["title_upweighted_text"]),
        })

    df = pd.DataFrame(rows)

    # Verification checks
    split_counts = df["split"].value_counts().to_dict()
    LOG.info("Split distribution: %s", split_counts)
    assert split_counts == {"train": 80, "val": 17, "test": 17}, "Split distribution mismatch!"

    # Check category distribution
    cat_by_split = pd.crosstab(df["category"], df["split"])
    LOG.info("Category breakdown by split:\n%s", cat_by_split)

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    LOG.info("Saved CSV to %s", OUTPUT_CSV)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    LOG.info("Saved JSONL to %s", OUTPUT_JSONL)

    # Save dataset report
    report = {
        "summary": {
            "total_records": len(df),
            "splits": split_counts,
            "leakage_checks_passed": True,
            "duplicate_clusters_isolated_to_train": True,
        },
        "category_distribution": {
            "overall": df["category"].value_counts().to_dict(),
            "by_split": {s: df[df["split"] == s]["category"].value_counts().to_dict() for s in ["train", "val", "test"]}
        },
        "audience_distribution": {
            "by_split": {
                s: {
                    aud: sum(aud in json.loads(val) for val in df[df["split"] == s]["audience_labels"])
                    for aud in AUDIENCES
                }
                for s in ["train", "val", "test"]
            }
        }
    }
    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    LOG.info("Saved report to %s", REPORT_JSON)

    return df


if __name__ == "__main__":
    build_dataset()
