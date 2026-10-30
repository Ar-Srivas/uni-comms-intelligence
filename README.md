## University Communications Intelligence

### Reproducible preprocessing

The raw dataset in `data/raw/` is an immutable input.  The preprocessing entry
point writes only to `data/processed/`:

```powershell
python scripts/preprocess.py
```

The command refuses to replace existing aggregate outputs.  Use `--force` to
regenerate them deterministically.  By default the final NLP dataset contains
only documents detected as English.  The master extracted, cleaned, quality,
and annotation outputs always retain every source PDF.  Use
`--include-non-english` to make non-English documents eligible for the final
dataset (documents with failed extraction or unresolved quality issues remain
out of it).

`scripts/preprocess.py --help` describes the optional input/output locations.
The quality report documents its thresholds and every exclusion/review reason.

### Category & Audience Classification

The canonical classification dataset is maintained at `data/processed/classification_dataset/classification_dataset.csv`.

#### Train Models
```bash
# Train Document Category Classifier (Single-label, 6 classes)
uv run python scripts/train_category_classifier.py

# Train Target Audience Classifier (Multi-label: students, faculty, administrators)
uv run python scripts/train_audience_classifier.py
```

#### Evaluate on Real Held-Out Test Set (24 Real Documents)
```bash
uv run python scripts/evaluate_classifiers.py
```

Evaluation outputs are generated under `reports/classification/`:
- `category_metrics.json` & `audience_metrics.json`
- `category_classification_report.txt` & `audience_classification_report.txt`
- `category_confusion_matrix.png`
- `category_errors.csv` & `audience_errors.csv`



### Web app

The FastAPI backend is in `src/`; the vanilla HTML, CSS, and JavaScript frontend
is in `frontend/`. At startup, the backend loads classifier artifacts from
`src/models/` and the summarizer from `src/models/summarizer_final/`.

Run the app from the repository root:

```powershell
uv run uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/` to use the frontend. The API has two routes:

| Method and route | Purpose |
| --- | --- |
| `GET /health` | Returns `{"status": "running"}` when the app is available. |
| `POST /process-notice` | Analyzes a notice and returns its title, predicted category and audience, extracted entities, persona summaries, and generated emails. |

Send `POST /process-notice` a JSON body with a document ID and notice text:

```json
{
  "doc_id": "notice_2026_001",
  "text": "Students must submit exam forms by 15/11/2026."
}
```
