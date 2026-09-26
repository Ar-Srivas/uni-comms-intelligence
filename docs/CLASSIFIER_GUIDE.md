# University Communications Intelligence: Category & Audience Classifier Guide

This guide documents the design, data pipeline, modeling strategy, evaluation results, error patterns, and inference workflows for the **Category & Audience Classification System** in `uni-comms-intelligence`.

---

## 1. Dataset Used
- **Canonical Dataset Location:** [`data/processed/classification_dataset/classification_dataset.csv`](file:///Users/adilsyed/Developer/Machine%20Learning/uni-comms-intelligence/data/processed/classification_dataset/classification_dataset.csv)
- **Dataset Size:** 150 total documents
  - **120 Real University Documents** (scraped from Lucknow, JNU, BHU, and IIT Bombay; 42 recovered from image-only scans using native Apple Vision OCR).
  - **30 Synthetic Documents** (conservative, realistic administrative notice templates generated to represent severely underrepresented classes and non-student roles).
- **Core Input Column:** `cleaned_text` (normalized via Unicode NFKC, header/footer margin stripping, hyphenation rejoin, and linebreak unwrapping).

---

## 2. Category Taxonomy
Six mutually exclusive document categories:
1. `admission`: Application windows, seat allocations, counselling rounds, mop-up schedules, eligibility lists, fee payment deadlines.
2. `academic`: Course registration, Add/Drop deadlines, commencement of classes, curriculum revisions, Ph.D. seminars, academic calendar schedules.
3. `examination`: Examination schedules, continuous assessment deadlines, invigilation duty rosters, answer script re-evaluation, UFM notifications, convocation medal lists.
4. `hostel`: Hall allocations, seniority lists, room vacation orders, mess committee regulations, dining tenders, residential rules.
5. `general`: University holiday circulars, national day celebrations, UGC anti-ragging compliance, sports selection trials, campus maintenance directives, administrative orders.
6. `announcements`: Code of conduct, university-wide IT maintenance advisories, campus health advisories, faculty association meetings, research showcase announcements.

---

## 3. Audience Taxonomy
Three target institutional recipient roles:
1. `students`: Enrolled undergraduate, postgraduate, and doctoral students, as well as prospective admission candidates and campus residents.
2. `faculty`: Teaching faculty, course instructors, research supervisors, department heads, and members of boards of studies.
3. `administrators`: Registrars, controllers of examinations, deans, section officers, provosts, finance officers, and estate personnel.

---

## 4. Why Category is Single-Label
Each official institutional circular is issued with a primary administrative intent and functional jurisdiction (e.g., an examination form notice issued by the Controller of Examinations is fundamentally `examination`, even though it affects students in an academic semester). Assigning exactly one primary category prevents jurisdiction ambiguity and enables clear routing in university portals.

---

## 5. Why Audience is Multi-Label
University notices routinely mandate action or compliance from multiple stakeholders simultaneously:
- An academic calendar or holiday declaration affects `["students", "faculty", "administrators"]`.
- A course Add/Drop extension or thesis presentation circular requires coordination between `["students", "faculty"]`.
- An examination results or disciplinary notification involves `["students", "administrators"]`.
- Statutory administrative directives and workload policies concern `["faculty", "administrators"]`.

Treating audience as multi-label ensures the system captures all intended recipients without forcing an artificial single-persona choice.

---

## 6. TF-IDF Representation
Both models use scikit-learn's `TfidfVectorizer`:
- **N-gram Range:** Unigrams and Bigrams `(1, 2)` to capture domain terminology (e.g., `"add drop"`, `"admit card"`, `"fee payment"`, `"end semester"`, `"hostel allotment"`).
- **Sublinear TF Scaling:** Applied to soften the impact of repeated administrative boilerplate words in long documents.
- **Frequency Filtering:** `min_df=2` (filters rare noise and typos) and `max_df=0.90` (filters ubiquitous tokens).
- **Stopwords:** Standard English stopwords removed.
- **Vocabulary Size:** ~13,780 extracted n-gram features.

---

## 7. Models Used
- **Category Classifier:**
  - Architecture: `TfidfVectorizer` $\to$ `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`
  - Decision Logic: Multi-class softmax with calibrated probability distribution across all 6 categories.
  - Baseline Comparison: Compared with `LinearSVC(C=1.0, class_weight='balanced')`. Logistic Regression was selected because it matches LinearSVC's validation accuracy (100%) while providing well-calibrated class probabilities.
- **Audience Classifier:**
  - Architecture: `TfidfVectorizer` $\to$ `OneVsRestClassifier(LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42))`
  - Decision Logic: Three independent binary classifiers (one per audience role) with calibrated sigmoid probabilities.
  - Multi-label Encoding: `MultiLabelBinarizer` with classes `['administrators', 'faculty', 'students']`.
  - Decision Threshold: Default $\tau = 0.50$ (with fallback to the highest confidence role if no label exceeds threshold).

---

## 8. Train / Validation / Test Methodology & Non-Leakage
To prevent data leakage, data partitioning was executed strictly on the **real documents first**:
- **Test Set:** **24 documents** (100% REAL, 0 synthetic).
- **Validation Set:** **18 documents** (100% REAL, 0 synthetic).
- **Training Set:** **108 documents** (78 real + 30 synthetic).

### Leakage Prevention Guarantees:
1. **Zero Test Contamination:** No synthetic records exist in test or validation sets.
2. **Zero ID Overlap:** Document IDs across train, val, and test are strictly disjoint.
3. **No Threshold Peeking:** All feature extraction, vectorizer vocabulary fitting, and hyperparameter tuning were conducted strictly on train and validation. The test set was touched only once for the final evaluation report.

---

## 9. Evaluation Metrics
Evaluated on the **24 real held-out test documents**:

### Category Metrics (Single-Label)
| Metric | Test Set Score |
| :--- | :---: |
| **Accuracy** | **95.83%** (23 / 24 correct) |
| **Macro Precision** | **0.9500** |
| **Macro Recall** | **0.9867** |
| **Macro F1-Score** | **0.9645** |
| **Weighted F1-Score** | **0.9606** |

#### Per-Category Performance:
| Category | Precision | Recall | F1-Score | Support (Real Test Docs) |
| :--- | :---: | :---: | :---: | :---: |
| `academic` | 1.0000 | 1.0000 | 1.0000 | 2 |
| `admission` | 1.0000 | 0.9333 | 0.9655 | 15 |
| `examination` | 1.0000 | 1.0000 | 1.0000 | 2 |
| `general` | 0.7500 | 1.0000 | 0.8571 | 3 |
| `hostel` | 1.0000 | 1.0000 | 1.0000 | 2 |
| `announcements`| 0.0000 | 0.0000 | 0.0000 | 0 *(only 1 real doc in entire raw corpus, placed in train)* |

---

### Audience Metrics (Multi-Label)
| Metric | Test Set Score |
| :--- | :---: |
| **Micro F1-Score** | **0.9375** |
| **Macro F1-Score** | **0.8786** |
| **Weighted F1-Score** | **0.9400** |
| **Exact-Match (Subset) Accuracy** | **83.33%** (20 / 24 notices exact match) |
| **Hamming Loss** | **0.0556** (only 4 binary decision errors across 72 total checks) |

#### Per-Audience Performance:
| Audience Role | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `students` | 0.9583 | 1.0000 | 0.9787 | 23 |
| `faculty` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `administrators` | 0.6667 | 1.0000 | 0.8000 | 4 |

---

## 10. Main Error Patterns

### Category Error Analysis (1 Error out of 24)
- **Doc 10 (`CIRC-2026-0010`):** *"Notice for Direct Admission under Sports Quota 2026-27"*
  - **True Category:** `admission`
  - **Predicted Category:** `general` (Confidence: 0.2993 vs. admission confidence: 0.1412)
  - **Root Cause:** Issued by the Athletic Association and Sports Pavilion of Lucknow University. The document is brief (293 characters) and shares strong sports vocabulary with trial notices (Docs 1 and 4), which are categorized as `general`.

### Audience Error Analysis (4 Errors out of 24)
- **Doc 3 (`CIRC-2026-0003`):** *"Strict Compliance with UGC Regulations 2009 for Prevention of Ragging"*
  - **True Audience:** `["administrators", "faculty", "students"]`
  - **Predicted Audience:** `["administrators", "students"]`
  - **Root Cause:** The model predicted `faculty` probability at `0.4976`, falling just 0.0024 short of the 0.50 threshold.
- **Doc 23 & Doc 30:** *"Commencement of classes for newly admitted students"*
  - **True Audience:** `["faculty", "students"]`
  - **Predicted Audience:** `["administrators", "faculty", "students"]`
  - **Root Cause:** Both notices include detailed bottom distribution blocks copying administrative authorities (*OSD to Vice-Chancellor, Joint Registrar, Evaluation Branch*), triggering the `administrators` classifier (confidences `0.5773` and `0.5852`).
- **Doc 33:** *"Implementation of Nasha Mukt Yuva for Viksit Bharat Campaign"*
  - **True Audience:** `["administrators", "faculty"]`
  - **Predicted Audience:** `["administrators", "faculty", "students"]`
  - **Root Cause:** The circular instructs departments to organize student pledges and youth activities, causing the student-level cue weights to push student confidence to `0.7251`.

---

## 11. How to Run Inference

### CLI Usage

```bash
# 1. Classify raw notice text directly
uv run python scripts/predict.py --text "All undergraduate students must register for End-Semester Examinations by October 15. The online fee payment portal is active. Late fees apply after the deadline."

# 2. Classify a saved text file
uv run python scripts/predict.py --file path/to/notice.txt

# 3. Classify a digital PDF directly
uv run python scripts/predict.py --pdf data/raw/pdfs/0029.pdf
```

### Python Module Usage

```python
from scripts.predict import NoticeClassifier, predict_notice

# One-off prediction helper
result = predict_notice("Meeting of the Board of Studies in Computer Science will be held on Monday at 10:30 AM to discuss syllabus revisions.")
print(result)

# Reusable classifier instance
classifier = NoticeClassifier(audience_threshold=0.5)
res = classifier.predict("Tentative seniority list for single seater boys hostel room allotment has been published.")
print(res["category"])  # -> 'hostel'
print(res["audience"])  # -> ['students']
```

### Example JSON Output

```json
{
  "category": "examination",
  "category_confidence": 0.221,
  "category_probabilities": {
    "academic": 0.1696,
    "admission": 0.1722,
    "announcements": 0.1433,
    "examination": 0.221,
    "general": 0.1451,
    "hostel": 0.1489
  },
  "audience": [
    "faculty",
    "students"
  ],
  "audience_confidences": {
    "administrators": 0.4996,
    "faculty": 0.5064,
    "students": 0.6026
  }
}
```

---

## 12. Reproducibility & Model Retraining

To reproduce the entire pipeline from scratch:

```bash
# 1. Rebuild canonical dataset with OCR recovery and stratified split
uv run python scripts/build_classification_dataset.py

# 2. Train Category Classifier
uv run python scripts/train_category_classifier.py

# 3. Train Audience Classifier
uv run python scripts/train_audience_classifier.py

# 4. Run test set evaluation and error analysis
uv run python scripts/evaluate_classifiers.py
```
