# Week 5 Group 11

For this week's submission, we incorporated prior feedback to refine and optimize our model logic (note: the dataset remains unchanged as we await additional university data). The three core components of our architecture are now fully unified into an end-to-end integrated pipeline. The pipeline dynamically loads predictions using trained binaries and model weights stored in our repository.

To ensure evaluation integrity, we implemented a dedicated data validation notebook that verifies clean, disjoint data splits a framework designed to seamlessly accommodate incoming datasets. To evaluate real world system robustness and edge case handling, we tested the pipeline on unseen emails sourced from our institutional Outlook accounts. Serving as an out of domain benchmark, this zero-shot dataset therefore offers a reliable assessment. 




## Technical Approach & Architecture

### 1. Document Classification (Stage 1)

* **Category Classification:** Utilizes a title-weighted TF-IDF vectorizer paired with a Logistic Regression classifier across five target categories (`academic`, `admission`, `examination`, `general`, `hostel`).
* **Audience Assignment:** Utilizes a multi-label Logistic Regression classifier with tuned decision thresholds per target audience class (`students`, `faculty`, `administrators`).

### 2. Information Extraction & NER (Stage 2)

* **Deterministic Matching:** Regular expressions extract structured entities, including explicit dates, times, contact emails, and embedded URLs.
* **Transformer-Based Extraction:** A `dslim/bert-base-NER` pipeline extracts dynamic target entities, specifically organizations (`ORG`) and personnel (`PERSON`).

### 3. Persona Summarization & Gating (Stage 3)

* **Conditioned Generation:** A fine-tuned `facebook/bart-base` model generates target audience summaries using persona-conditioned task prefixes (`summarize for student:`, `summarize for faculty:`).
* **Audience-Gated Fallback:** Pipeline routing verifies target audience tags emitted by Stage 1. If an audience tag is absent for a target persona, the pipeline bypasses text generation and outputs a deterministic null response (`"No specific [persona] action stated."`) to enforce factual alignment.

---

## Dataset Integrity & Split Protocol

The system dataset consists of **150 total documents** (120 real institutional notices and 30 synthetic data augmentations):

| Dataset Split | Real Notices | Synthetic / Augmented | Total Documents | Split Integrity Standard |
| --- | --- | --- | --- | --- |
| **Train** | 78 | 30 | 108 | Zero document ID overlap |
| **Validation** | 18 | 0 | 18 | Isolated validation set |
| **Test** | 24 | 0 | 24 | Held-out evaluation set |
| **Total** | **120** | **30** | **150** | **100% Disjoint Verification** |

* **Zero Leakage:** Document splits maintain complete separation across all stages, ensuring synthetic data is restricted exclusively to the training split.
* **Persona Cohesion:** Multi-persona views associated with a single document are held strictly within the same evaluation split.

---

## Quantitative Performance Benchmarks

### Stage 1: Classification Performance

Evaluated on the held-out test dataset ($n=24$):

* **Category Classification Accuracy:** `0.9167`
* **Category Macro-F1:** `0.8764`
* **Audience Assignment Micro-F1:** `0.8852`
* **Audience Assignment Macro-F1:** `0.8946`

### Stage 2: Named Entity Recognition (NER)

Evaluated against gold-annotated evaluation notice snippets ($n=15$):

* **Exact Match Metrics:** Precision: `0.5250` | Recall: `0.7500` | F1 Score: `0.6176`
* **Overlap Match Metrics:** Precision: `0.6750` | Recall: `0.9286` | F1 Score: `0.7817`
* **Entity F1 Scores (Overlap Match):**
* `DATE`: 1.0000
* `TIME`: 1.0000
* `EMAIL`: 1.0000
* `ORG`: 0.6087
* `PERSON`: 0.5714



### Stage 3: Persona Summarization Metrics

Evaluated on held-out test notices across 30 generated summaries:

* **ROUGE-1:** `0.2470`
* **ROUGE-2:** `0.1225`
* **ROUGE-L:** `0.2031`
* **Null Fallback Rate:** `50.0%`

---

## Robustness & Perturbation Evaluation

Pipeline performance evaluation under varied input text transformations ($n=24$ test notices):

| Input Perturbation | Stage 1 Category Accuracy | Stage 1 Audience F1 | Stage 2 NER F1 (Overlap) | Stage 3 ROUGE-L |
| --- | --- | --- | --- | --- |
| **Clean Baseline** | 0.9167 | 0.8852 | 0.8172 | 0.1718 |
| **Lowercase Text** | 0.9167 | 0.8852 | 0.7516 | 0.1650 |
| **Uppercase Text** | 0.9167 | 0.8852 | 0.6960 | — |
| **OCR Noise (~2%)** | 0.9167 | 0.9032 | 0.7557 | 0.1764 |
| **Linebreak Ingestion** | 0.9167 | 0.8852 | 0.8172 | 0.2698 |
| **50% Truncation** | 0.8750 | 0.8710 | 0.4825 | 0.1865 |
| **Email Wrapper** | 0.9583 | 0.8710 | 0.7290 | 0.2118 |

---

## Error Modes & Pipeline Limitations

1. **Classification Cascade Errors:** Misclassifications in Stage 1 multi-label audience prediction directly trigger Stage 3 null gating, suppressing summary generation for relevant audience personas.
2. **Subword WordPiece Segmentation:** Subword tokenization within `dslim/bert-base-NER` fragments uncommon domain vocabulary and proper names into subword artifacts, reducing exact-match entity accuracy.
3. **Sequence Truncation Vulnerability:** Document truncation severely reduces Stage 2 entity recall due to key entity position loss.

## Future Development Plan

To transition the system into a deployable solution, future work will focus on enhancing model logic, expanding data coverage, and implementing a production-ready architecture without relying on large language models or high-overhead SOTA architectures:

* **Model Optimization & Feature Engineering:** Refine Stage 1 classification using title-weighted TF-IDF feature selection and hyperparameter-tuned SVM/Logistic Regression classifiers. Upgrade Stage 3 summarizers by tuning lightweight BART/T5 models with strict constrained decoding to eliminate hallucinated dates or URLs.


* **Joint Entity-Relation Extraction:** Upgrade Stage 2 from isolated token-level NER to rule-augmented joint entity-relation extraction, automatically pairing detected deadlines directly with their corresponding administrative roles and required actions.


* **Dataset Scaling & Active Learning:** Ingest and annotate multi-departmental university data as it arrives to expand coverage beyond the current 150-document set, establishing an active learning feedback loop to flag low-confidence predictions for human review.


* **Microservices & Webhook Integration:** Containerize the multi-stage pipeline using Docker and FastAPI, deploying webhooks to ingest incoming circulars directly from institutional Outlook APIs into an interactive, role-based front-end dashboard.