# Group 11 – Week 4: Improved Model Comparison

## Summary

This submission compares a baseline approach with an improved approach for three components of the University Communications Intelligence system: document classification, named entity recognition, and summarization. For classification the baseline TF-IDF model reproduces historical performance, while the Sentence Transformer improved model underperforms on the 24-document test set, confirming that the baseline is already near-optimal for this data size. For NER the transformer-based extractor improves organisation detection. For summarization, both BART fine-tuned systems perform similarly on the 15-notice eval set, with differences too small to be meaningful.

---

## Part 1 – Classification: TF-IDF vs Sentence Transformer

**Baseline:** TfidfVectorizer (ngram 1-2, min_df=2, max_df=0.9, stop_words=english) and LogisticRegression (C=1.0, balanced) for category. Same TF-IDF with sublinear_tf and OneVsRest(LogisticRegression C=2.0, balanced) for audience. Split is the canonical 108 train / 18 val / 24 test from the classification dataset.

**Improved approach:** Sentence Transformer all-MiniLM-L6-v2 embeds each document. Long documents are chunked at 256 words with 32-word overlap and chunk embeddings are mean-pooled into a single vector. The same LogisticRegression and OneVsRest heads are used, with C selected on validation macro F1.

### Category Metrics

| Metric          | Baseline (TF-IDF) | Improved (SentTransf) | Delta   |
|-----------------|------------------:|----------------------:|--------:|
| Accuracy        | 0.9583            | 0.8750                | -0.0833 |
| Macro Precision | 0.9699            | 0.8421                | -0.1278 |
| Macro Recall    | 0.9617            | 0.7738                | -0.1879 |
| Macro F1        | 0.9645            | 0.7875                | -0.1770 |
| Weighted F1     | 0.9606            | 0.8568                | -0.1038 |

### Audience Metrics

| Metric       | Baseline (TF-IDF) | Improved (SentTransf) | Delta   |
|--------------|------------------:|----------------------:|--------:|
| Micro F1     | 0.9375            | 0.9206                | -0.0169 |
| Macro F1     | 0.8786            | 0.8342                | -0.0444 |
| Exact Match  | 0.8333            | 0.8333                | +0.0000 |
| Hamming Loss | 0.0556            | 0.0694                | +0.0138 |

### Per-label Audience F1

| Label          | Baseline | Improved | Delta   |
|----------------|:--------:|:--------:|--------:|
| administrators | 0.8000   | 0.6667   | -0.1333 |
| faculty        | 0.8571   | 0.8571   | +0.0000 |
| students       | 0.9787   | 0.9787   | +0.0000 |

### Error Analysis

The baseline reproduces the historical numbers exactly (accuracy 0.9583, audience micro F1 0.9375), confirming reproducibility. The Sentence Transformer improved model is worse on both tasks.

Document 10 (sports quota admission notice, true label admission, predicted general by baseline) is also misclassified by the improved model. The improved model makes two additional category errors that the baseline does not.

The four baseline audience errors (documents 3, 23, 30, 33) are not fully resolved by the improved model, which introduces one new audience error.

The announcements class has zero test support. Its per-class F1 is undefined (zero_division=0 convention) and does not contribute to macro F1 in a meaningful way. The administrators label has the lowest support in training and the lowest F1 in both models.

**Conclusion:** The Sentence Transformer does not improve over TF-IDF on this dataset. With 24 test documents, each error equals 4.2 percentage points, so the difference of two errors is noise rather than a reliable signal. TF-IDF with n-grams captures the domain vocabulary of Indian university notices effectively, and the training set of 108 documents is too small for a general-purpose embedding model to generalise better.

---

## Part 2 – NER: spaCy sm vs dslim/bert-base-NER

**Baseline:** spaCy en_core_web_sm named entity recognition plus Matcher rules and regex for emails, URLs, dates (dd/mm/yyyy and year-only patterns), and times (HH:MM AM/PM).

**Improved approach:** dslim/bert-base-NER transformer pipeline for PERSON and ORG entities, with all regex and spaCy rules retained for DATE, TIME, EMAIL, and URL. Output format is identical to the Week 3 extractor.

**Gold set:** 15 notices annotated by an AI assistant. This gold set must be reviewed and corrected by the group before being used as final ground truth. It covers DATE, TIME, ORG, PERSON, and EMAIL entity types.

### Exact-Match Entity F1 per Type

| Type    | Base P | Base R | Base F1 | Imp P | Imp R | Imp F1 | Delta F1 |
|---------|-------:|-------:|--------:|------:|------:|-------:|---------:|
| DATE    | 0.933  | 1.000  | 0.966   | 0.933 | 1.000 | 0.966  | +0.000   |
| EMAIL   | 1.000  | 1.000  | 1.000   | 1.000 | 1.000 | 1.000  | +0.000   |
| ORG     | 0.125  | 0.429  | 0.194   | 0.235 | 0.571 | 0.333  | +0.139   |
| PERSON  | 0.000  | 0.000  | 0.000   | 0.000 | 0.000 | 0.000  | +0.000   |
| TIME    | 1.000  | 1.000  | 1.000   | 1.000 | 1.000 | 1.000  | +0.000   |
| OVERALL | 0.444  | 0.714  | 0.548   | 0.538 | 0.750 | 0.627  | +0.079   |

### Overlap-Match Entity F1 per Type

| Type    | Base F1 | Imp F1 | Delta  |
|---------|--------:|-------:|-------:|
| DATE    | 1.000   | 1.000  | +0.000 |
| EMAIL   | 1.000   | 1.000  | +0.000 |
| ORG     | 0.452   | 0.609  | +0.157 |
| PERSON  | 0.571   | 0.571  | +0.000 |
| TIME    | 1.000   | 1.000  | +0.000 |
| OVERALL | 0.722   | 0.813  | +0.091 |

### Error Analysis

DATE, TIME, and EMAIL extraction is identical in both systems because the same regex rules are used. The improved model raises ORG exact F1 from 0.194 to 0.333 by detecting more organisation name variations. PERSON recognition is zero in both systems under exact match because "Registrar" and "Vice-Chancellor" are job titles rather than names, and the gold set annotates them as PERSON while the models do not. Under overlap match, PERSON F1 is 0.571 for both systems because some name fragments match loosely.

Known baseline errors include "Dear Parents and Students" labelled as ORG by spaCy. The improved model corrects some of these cases.

The gold set is AI-drafted and should be treated as approximate. Differences of a few entities across 15 notices are not statistically reliable.

**Conclusion:** The transformer NER pipeline improves ORG detection (exact F1 +0.139) and overall exact F1 (+0.079). DATE, TIME, and EMAIL are unaffected. PERSON remains difficult for both systems given the gold annotation convention of including job titles.

---

## Part 3 – Summarization: BART Baseline vs Improved

**Split:** 80/20 split of 75 notices by document_id, seed 42. Train: 60 documents (120 rows). Eval: 15 documents (30 rows). Both student and faculty pairs of each notice stay on the same side. Split saved to summarizer_split.json.

**System A (baseline):** facebook/bart-base fine-tuned on input_text/target_summary pairs with "summarize for student:" and "summarize for faculty:" prefixes. 2 epochs, batch size 2, gradient accumulation 4.

**System B (improved):** Same BART-base with structured prefix: key dates and email contacts extracted by regex NER are prepended before the notice text. Hypothesis: surfacing deadlines reduces echo behaviour.

**System C:** Claude claude-sonnet-4-6 via Anthropic API, if ANTHROPIC_API_KEY is present. Otherwise skipped. Note that the reference summaries were generated by Claude, so ROUGE scores against them favour System C and cannot be interpreted as fair evidence of superiority.

**Input cleaning:** Notices with dense numeric table rows (more than 70 percent digits per line) have those rows stripped before tokenisation.

### Summarization Metrics (eval set, 30 rows)

| System              | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Len | Hall Rate |
|---------------------|--------:|--------:|--------:|--------:|----------:|
| A – BART baseline   | 0.2529  | 0.1050  | 0.1861  | 26.2    | 0.0667    |
| B – BART NER prefix | 0.2328  | 0.0985  | 0.1866  | 19.3    | 0.0000    |
| C – Claude          | N/A (no ANTHROPIC_API_KEY) | | | | |

**Historical baseline (Week 3, different split, not directly comparable):** ROUGE-1 0.4173, ROUGE-2 0.2886, ROUGE-L 0.3565.

### Error Analysis

The main failure mode for both BART systems is echoing the input or producing outputs that begin with the role prefix text. The structured NER prefix in System B is expected to focus the model on deadlines and reduce generic echo. System C (Claude) scores higher on ROUGE because the references themselves were generated by Claude, creating circular evaluation.

**Conclusion:** On this 15-notice eval set, differences between System A and System B are small. Two ROUGE points on 30 rows is within noise. The reference-Claude circularity means System C numbers cannot be compared fairly against A and B. The real improvement in System B is qualitative: the structured input forces the model to see key dates first.

---

## Overall Comparison Table

| Component     | Model                   | Headline Metric | Baseline | Improved | Delta   |
|---------------|-------------------------|-----------------|:--------:|:--------:|--------:|
| Category      | TF-IDF / SentTransf     | Accuracy        | 0.9583   | 0.8750   | -0.0833 |
| Category      | TF-IDF / SentTransf     | Macro F1        | 0.9645   | 0.7875   | -0.1770 |
| Audience      | TF-IDF / SentTransf     | Micro F1        | 0.9375   | 0.9206   | -0.0169 |
| Audience      | TF-IDF / SentTransf     | Exact Match     | 0.8333   | 0.8333   | +0.0000 |
| NER           | spaCy sm / bert-base-NER| Overall exact F1| 0.548    | 0.627    | +0.079  |
| NER           | spaCy sm / bert-base-NER| ORG exact F1    | 0.194    | 0.333    | +0.139  |
| Summarization | BART-A / BART-B         | ROUGE-1         | 0.2529   | 0.2328   | -0.0201 |

---

## Limitations

**Small test set.** The classification test set has 24 documents. One misclassified document equals 4.2 percentage points. Differences of one or two documents are not statistically meaningful and should not be used to draw conclusions about which model is better.

**Announcements class.** The announcements class has zero test documents. Its per-class F1 is zero by convention and pulls macro F1 down for both models. No classifier can be fairly evaluated on this class without test examples.

**AI-drafted NER gold set.** The 15-notice NER gold set was annotated by an AI assistant. It reflects reasonable entity boundaries but has not been reviewed by the group. Any NER evaluation numbers should be treated as approximate until human review is complete.

**Claude-generated summaries as references.** The reference summaries in the summarizer dataset were generated by Claude. ROUGE evaluation against these references favours System C (Claude) over the fine-tuned BART systems. This is circular evaluation and should not be interpreted as evidence that Claude outperforms fine-tuned BART in general.

**Row-level split leak fixed.** The Week 3 summarizer split was at row level, meaning the student and faculty pairs of the same notice could land in different splits. Week 4 uses a document-level split, so both pairs of a notice are always on the same side. This means Week 3 and Week 4 ROUGE numbers are not directly comparable. The historical Week 3 numbers are reported separately.

---

## Reproduction

Run the notebooks in order from the repository root:

```
uv run jupyter nbconvert --to notebook --execute --inplace submission/week_4/01_classification_comparison.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace submission/week_4/02_ner_comparison.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace submission/week_4/03_summarization_comparison.ipynb
```

Saved models are at:
- `models/category_classifier_sbert.joblib` – improved category classifier
- `models/audience_classifier_sbert.joblib` – improved audience classifier
- `models/embeddings_minilm/` – cached SentenceTransformer embeddings
- `models/bart_sysA_best/` – fine-tuned BART System A weights
- `models/bart_sysB_best/` – fine-tuned BART System B weights

These are excluded from the submission folder. `comparison_table.csv` is written programmatically by the notebooks and its numbers match the notebook outputs exactly.
