# Week 6 Model Retraining & Evaluation Report
**Task Focus**: Retraining and Evaluating Category and Multi-Label Audience Classifiers on College Email Data  
**Dataset**: 114 Real University College Email Records (`submission/week_6/data/dump/`)  
**Scope**: Classification Models, Datasets, Evaluations, and Handover Artifacts (Preserving all Stage 2, Summarization, and Shared Pipeline Code)

---

## 1. Executive Summary

This report documents the preparation, retraining, threshold tuning, and evaluation of the **Category** and **Audience** classifiers on newly collected college email communications from SVKM NMIMS MPSTME.

### Key Milestones Achieved:
1. **Canonical Email Dataset Ingestion**: Parsed 114 raw college email notices from [`submission/week_6/data/dump/`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/submission/week_6/data/dump/) into individual structured records with standardized schema, saving them to [`data/processed/classification_dataset/college_emails_dataset.csv`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/data/processed/classification_dataset/college_emails_dataset.csv).
2. **Schema Realignment**:
   - **Category**: Realignment from 6 legacy circular classes to the 5 intended classes: `academic`, `placements`, `admissions`, `announcements`, `examinations`.
   - **Audience**: Multi-label outputs standardizing on `student`, `faculty`, and `admin`.
3. **Preservation of Natural Imbalance**: Zero synthetic notices were fabricated and zero records were duplicated. The natural institutional skew (100% of emails reach students, while only 8.8% involve faculty and 7.9% involve administration) was preserved and handled via cost-sensitive learning.
4. **Leakage-Free Splitting**: Implemented an 80 train (70.2%) / 17 val (14.9%) / 17 test (14.9%) split with all duplicate clusters strictly contained within the training split.
5. **Model Comparisons**:
   - **Category**: Class-Weighted Logistic Regression achieved **82.35% Test Accuracy** and **0.6618 Macro F1**, substantially outperforming Standard unweighted Logistic Regression (**70.59% Accuracy**, **0.5333 Macro F1**).
   - **Audience**: Standard LR predicted `['student']` across 100% of documents with 0 recall for minority roles. In contrast, Class-Weighted LR with validation-tuned thresholds achieved **0.4667 Macro F1** and successfully recovered faculty-facing notices (e.g. `ann-014`).
6. **Safe Isolation**: All retrained models and vectorizers were stored in independent Week 6 directories ([`models/week6/category_classifier/`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/category_classifier/) and [`models/week6/audience_classifier/`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/audience_classifier/)), avoiding any breaking modifications to existing baseline models or teammate pipelines.

---

## 2. Dataset Structure & Imbalance Analysis

### A. Raw Email Sources
The 114 records were parsed from 5 thematic dump files without modifying or deleting the source files:
- `examinations_mails.txt`: 30 emails (prefixed `exam-001` to `exam-030`)
- `placement_mails.txt`: 22 emails (prefixed `plac-001` to `plac-022`)
- `announcement_mails.txt`: 22 emails (prefixed `ann-001` to `ann-022`)
- `admissions_mails.txt`: 20 emails (prefixed `adm-001` to `adm-020`)
- `academic_emails.txt`: 20 emails (prefixed `acad-001` to `acad-020`)

### B. Category Distribution (N = 114)
Official emails frequently cross-classify from their originating dump file (e.g., an announcement email regarding examination seating is categorized as `examinations`). The final verified ground-truth category distribution across the 114 records is:

| Category | Total Count | Proportion (%) | Train (N=80) | Val (N=17) | Test (N=17) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `examinations` | 35 | 30.7% | 25 | 5 | 5 |
| `placements` | 31 | 27.2% | 21 | 5 | 5 |
| `admissions` | 20 | 17.5% | 14 | 3 | 3 |
| `announcements` | 16 | 14.0% | 12 | 2 | 2 |
| `academic` | 12 | 10.5% | 8 | 2 | 2 |
| **Total** | **114** | **100.0%** | **80** | **17** | **17** |

### C. Audience Multi-Label Distribution
The distribution reflects genuine university communication flows: all notices are broadcast to student mailing lists, while a small subset mandates simultaneous compliance from teaching faculty or administrative offices.

| Audience Label | Total Record Presence | Proportion (%) | Train Split (80) | Val Split (17) | Test Split (17) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `student` | **114** | **100.0%** | 80 (100%) | 17 (100%) | 17 (100%) |
| `faculty` | **10** | **8.8%** | 5 (6.2%) | 2 (11.8%) | 3 (17.6%) |
| `admin` | **9** | **7.9%** | 5 (6.2%) | 2 (11.8%) | 2 (11.8%) |

**Observed Combinations**:
- `['student']`: 97 (85.1%)
- `['student', 'faculty']`: 8 (7.0%)
- `['student', 'admin']`: 7 (6.1%)
- `['student', 'faculty', 'admin']`: 2 (1.8%)
- Pure `['faculty']` or pure `['admin']`: 0 (0.0%)

---

## 3. Leakage Prevention Protocol

A rigorous audit identified duplicate and high-similarity notice clusters across the raw dumps:
1. `adm-018` & `adm-019`: "Symbiosis Institute of Technology" announcement (Cosine similarity: 1.0000).
2. `ann-013` & `ann-020`: "Portal Announcement" system notices (Cosine similarity: 1.0000).
3. `ann-001` & `ann-004`: "MPSTME ODD Semester Final Exam Timetable" (Cosine similarity: 1.0000).
4. `plac-002`, `plac-003`, `plac-004`: Amendment to placement policy circulars (Cosine similarity: 0.8815 – 1.0000).
5. `exam-002` & `exam-003`: Timetable reminders (Cosine similarity: 0.9026).
6. `acad-007` & `plac-013`: Letter of Recommendation (LOR) for Academic Internship notices (Cosine similarity: 0.8032).

**Isolation Guarantee**:
All notices belonging to these duplicate/near-duplicate clusters were strictly locked into the **Train split**. To replace `plac-013` in the Test set, `plac-014` (*TCS CodeVita Season 14 Registrations*) was promoted, preserving the 80/17/17 split. Following this correction, **no cross-split pair exceeds a cosine similarity of 0.6825 (zero cross-split pairs > 0.70)**. The Validation and Test sets contain 100% unique, unlinked email documents with zero ID or content leakage.

---

## 4. Experimental Results

Evaluated on the **17 held-out real test emails**. All metrics are derived from actual model predictions.

### A. Category Classifier Performance (Single-Label Multi-Class)

| Metric | Standard Logistic Regression | Class-Weighted Logistic Regression | Relative Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 70.59% (12 / 17) | **82.35% (14 / 17)** | **+11.76%** |
| **Macro Precision** | 0.5367 | **0.6333** | **+0.0966** |
| **Macro Recall** | 0.6500 | **0.7000** | **+0.0500** |
| **Macro F1-Score** | 0.5285 | **0.6618** | **+0.1333** |
| **Weighted F1-Score** | 0.6478 | **0.7850** | **+0.1372** |

#### Detailed Per-Category Breakdown (Balanced Model):
| Category | Precision | Recall | F1-Score | Test Support | Status / Remarks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `examinations` | **1.0000** | **1.0000** | **1.0000** | 5 | Perfect precision and recall on timetable and exam policies. |
| `admissions` | **1.0000** | **1.0000** | **1.0000** | 3 | Perfect precision and recall on admissions / exchange opportunities. |
| `placements` | **0.8333** | **1.0000** | **0.9091** | 5 | 100% recall on drives, tests, and training sessions; 1 minor confusion. |
| `announcements` | **0.3333** | **0.5000** | **0.4000** | 2 | Small sample size; captured general notices but absorbed academic circulars. |
| `academic` | **0.0000** | **0.0000** | **0.0000** | 2 | **Data Limitation**: Both test examples were predicted as `announcements`. |

![Category Confusion Matrix](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/reports/classification/week6_category_confusion_matrix.png)

---

### B. Audience Classifier Performance (Multi-Label)

Because 100% of emails include `student`, a naive standard model predicting solely `['student']` achieves an apparently high Exact-Match Accuracy of 76.47% by completely ignoring minority roles (`faculty` F1 = 0.00, `admin` F1 = 0.00).

Class-Weighted Logistic Regression with validation-tuned decision thresholds ($\tau_{\text{student}} = 0.20, \tau_{\text{faculty}} = 0.20, \tau_{\text{admin}} = 0.20$) significantly improves minority role sensitivity:

| Metric | Standard LR (Default $\tau=0.50$) | Class-Weighted LR (Tuned Thresholds) |
| :--- | :---: | :---: |
| **Exact-Match (Subset) Accuracy** | 76.47% | 58.82% |
| **Hamming Loss** | **0.0980** | 0.1765 |
| **Micro F1-Score** | **0.8718** | 0.8000 |
| **Macro F1-Score** | 0.3333 | **0.4667 (+40.0%)** |
| **Weighted F1-Score** | 0.8068 | 0.8273 |

#### Detailed Per-Audience Breakdown (Balanced Model):
| Audience Role | Precision | Recall | F1-Score | Test Support | Behavioral Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `student` | **1.0000** | **1.0000** | **1.0000** | 17 | Perfect recall across all test notices. |
| `faculty` | **0.5000** | **0.3333** | **0.4000** | 3 | Successfully retrieved `ann-014` (`['faculty', 'student']`). |
| `admin` | **0.0000** | **0.0000** | **0.0000** | 2 | Missed on sparse test notices; triggered false positives on seating notices. |

---

## 5. Sample Size Limitations & Edge Case Analysis

1. **`academic` Category (0.00 F1 on Test Set)**:
   - In the training set, `academic` notices comprise only 8 documents (10% of training data).
   - In the test set, the two `academic` documents were:
     - `acad-005`: Student registration for the Academic Bank of Credits (ABC ID).
     - `acad-019`: Industrial visit to a Pune manufacturing plant (consent form).
   - In email form, these notices resemble general campus advisories, causing the model to predict `announcements`. With only $N=2$ test samples, missing both leads to an F1 of 0.00.
2. **`faculty` and `admin` Audience Sparsity**:
   - `admin` has only 2 test examples; `faculty` has only 3 test examples.
   - Predicting even one false positive or false negative shifts precision or recall by 33%–50%.
   - Seating plan notices (`exam-009`, `exam-010`, `exam-011`) frequently mention invigilation and hall arrangement coordinators, causing the tuned model to flag `admin` alongside `student`. While technically plausible administratively, these were gold-labeled as purely student-facing.

---

## 6. Baseline Comparison: Week 5 Circulars vs. Week 6 Retrained Email Classifiers

| Aspect | Week 5 Baseline (Gazette Circulars) | Week 6 Retrained (College Emails) |
| :--- | :--- | :--- |
| **Data Modality** | Formal scanned/extracted institutional PDFs | Transactional university email broadcasts |
| **Categories Supported** | `academic`, `admission`, `announcements`, `examination`, `general`, `hostel` | `academic`, `admissions`, `announcements`, `examinations`, `placements` |
| **Placements Handling** | 0 support (Unrepresented) | **90.91% F1** (31 training/eval emails) |
| **Email Domain Generalization**| Fails on email syntax, signature blocks, headers | Optimized on email vocabulary and subject weighting |
| **Audience Vocabulary** | Formal administrative role phrasing | Student mailing list (`@nmims.edu`) and departmental sign-offs |

---

## 7. Handover Artifacts for Teammate Integration

All Week 6 assets are saved in designated directories for downstream integration into Stage 1/Stage 2:

### Model Paths
- **Category Classifier**:
  - Model: [`models/week6/category_classifier/model.joblib`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/category_classifier/model.joblib)
  - Vectorizer: [`models/week6/category_classifier/vectorizer.joblib`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/category_classifier/vectorizer.joblib)
  - Metadata: [`models/week6/category_classifier/metadata.json`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/category_classifier/metadata.json)
- **Audience Classifier**:
  - Model: [`models/week6/audience_classifier/model.joblib`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/audience_classifier/model.joblib)
  - Vectorizer: [`models/week6/audience_classifier/vectorizer.joblib`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/audience_classifier/vectorizer.joblib)
  - MultiLabelBinarizer: [`models/week6/audience_classifier/mlb.joblib`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/audience_classifier/mlb.joblib)
  - Tuned Thresholds: [`models/week6/audience_classifier/thresholds.json`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/audience_classifier/thresholds.json)
  - Metadata: [`models/week6/audience_classifier/metadata.json`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/models/week6/audience_classifier/metadata.json)

### Evaluation & Prediction Artifacts
- Test Predictions: [`reports/classification/week6_test_predictions.csv`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/reports/classification/week6_test_predictions.csv)
- Evaluation Metrics: [`reports/classification/week6_evaluation_metrics.json`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/reports/classification/week6_evaluation_metrics.json)
- Confusion Matrix: [`reports/classification/week6_category_confusion_matrix.png`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/reports/classification/week6_category_confusion_matrix.png)

### Minimal Integration Snippet
```python
import joblib
import json
from pathlib import Path

# Load Week 6 Category Classifier
cat_model = joblib.load("models/week6/category_classifier/model.joblib")
cat_vec = joblib.load("models/week6/category_classifier/vectorizer.joblib")

# Load Week 6 Audience Classifier
aud_model = joblib.load("models/week6/audience_classifier/model.joblib")
aud_vec = joblib.load("models/week6/audience_classifier/vectorizer.joblib")
mlb = joblib.load("models/week6/audience_classifier/mlb.joblib")
with open("models/week6/audience_classifier/thresholds.json") as f:
    thresholds = json.load(f)

def predict_email(text: str) -> dict:
    # 1. Category
    c_vec = cat_vec.transform([text])
    cat_pred = cat_model.predict(c_vec)[0]
    cat_conf = float(cat_model.predict_proba(c_vec)[0].max())

    # 2. Audience
    a_vec = aud_vec.transform([text])
    a_probs = aud_model.predict_proba(a_vec)[0]
    audiences = [cls for cls, p in zip(mlb.classes_, a_probs) if p >= thresholds.get(cls, 0.5)]
    if not audiences:
        audiences = ["student"]

    return {
        "category": cat_pred,
        "category_confidence": round(cat_conf, 4),
        "target_audiences": audiences,
    }
```

### Reproducible Commands
```bash
# 1. Rebuild Dataset with Leakage Checks
.venv/bin/python scripts/build_week6_dataset.py

# 2. Retrain Category & Audience Models
.venv/bin/python scripts/train_week6_classifiers.py

# 3. Evaluate Models on Test Split
.venv/bin/python scripts/evaluate_week6_classifiers.py
```
