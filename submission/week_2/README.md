# Week 2 — Dataset & Baseline

## Project
**University Communications Intelligence**

## Objective
Prepare the university communications dataset and establish working baseline models for:
1. Document category classification
2. Target audience classification
3. Persona-driven notice summarization

## Completed
* Data collection from public university portals
* Text extraction and OCR recovery
* Text cleaning and normalization
* Category and audience annotation
* Leakage-free train / validation / test partitioning
* Category classification baseline & initial evaluation
* Audience classification baseline & initial evaluation
* Summarization baseline & initial evaluation

## Dataset
Maintained at `data/processed/classification_dataset/classification_dataset.csv`:
* **150 total documents**
* **108 training documents** (78 real + 30 synthetic)
* **18 validation documents** (100% real)
* **24 test documents** (100% real)

Synthetic documents are restricted strictly to the training split. Validation and test sets contain only real university communications.

## Category Classes (Single-Label)
- `admission`
- `academic`
- `examination`
- `hostel`
- `general`
- `announcements`

## Audience Classes (Multi-Label)
- `students`
- `faculty`
- `administrators`

## Baseline Models

### Category Classification
- **Architecture:** TF-IDF $(1, 2)$-grams + Logistic Regression (`class_weight="balanced"`)

### Audience Classification
- **Architecture:** TF-IDF $(1, 2)$-grams + One-vs-Rest Logistic Regression (`class_weight="balanced"`)

### Summarization
- **Architecture:** `facebook/bart-base` fine-tuned for role-conditioned persona summarization (`summarize for student:` / `summarize for faculty:`)

## Initial Evaluation Results

### Category Classification (Test Set: 24 Real Notices)
- **Accuracy:** 95.83% (23 / 24 correct)
- **Macro Precision:** 0.9500
- **Macro Recall:** 0.9867
- **Macro F1-Score:** 0.9645
- **Weighted F1-Score:** 0.9606

*(Note: The `announcements` category has support = 0 in the test set because the single real announcement in the corpus was allocated to training; its test performance cannot be assessed).*

### Audience Classification (Test Set: 24 Real Notices)
- **Micro F1-Score:** 0.9375
- **Macro F1-Score:** 0.8786
- **Weighted F1-Score:** 0.9400
- **Exact-Match Accuracy:** 83.33% (20 / 24 exact multi-label sets)
- **Hamming Loss:** 0.0556

Per-Audience breakdown:
| Audience Role | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| `students` | 0.9583 | 1.0000 | 0.9787 |
| `faculty` | 1.0000 | 0.7500 | 0.8571 |
| `administrators` | 0.6667 | 1.0000 | 0.8000 |

### Summarization (Evaluation Set: 5 Labeled Pairs)
- **ROUGE-1:** 30.76% (0.3076)
- **ROUGE-2:** 19.62% (0.1962)
- **ROUGE-L:** 28.15% (0.2815)
- **ROUGE-Lsum:** 28.15% (0.2815)
- **Date Hallucination Check:** Zero date hallucinations detected using spaCy entity extraction.

## Notebook Guide

```text
01_data_preprocessing.ipynb
→ Data collection, text extraction, OCR recovery, cleaning, annotation, dataset health

02_baseline_models.ipynb
→ Category + audience classification baselines, metrics, confusion matrix, error analysis

03_summarizer.ipynb
→ Persona-based summarization baseline (BART-base) and ROUGE evaluation
```

## Reproducibility
The notebooks load the canonical dataset from:
`data/processed/classification_dataset/classification_dataset.csv`

All code is fully executable and reproducible with `RANDOM_STATE = 42`.

## Repository
The complete project implementation, source PDFs, OCR scripts, trained models, and detailed audit reports remain in the main repository.
