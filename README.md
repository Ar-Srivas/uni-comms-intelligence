# University Communications Intelligence

## Run the app

From the repository root, install the dependencies and start the server:

```powershell
uv sync
uv run uvicorn src.api:app --reload
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

## Project layout

- `frontend/` contains the web interface:
  - `index.html` — page structure
  - `styles.css` — page styling
  - `script.js` — browser-side behavior and API requests
- `src/` contains the backend:
  - `api.py` — FastAPI app, routes, and frontend serving
  - `integrated_pipeline.py` — notice classification, entity extraction, and summaries
  - `emails.py` — builds audience-specific emails from pipeline results
  - `models/` — classifier and summarizer model files loaded by the app
  - `data/` — pipeline data and intermediate outputs

## API

Interactive API documentation is available at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

- `GET /health` checks that the app is running.
- `POST /process-notice` analyzes notice text and returns its title, category,
  audience, extracted entities, summaries, and generated emails.

Send `POST /process-notice` a JSON body with a document ID and notice text:

```json
{
  "doc_id": "notice_2026_001",
  "text": "Students must submit exam forms by 15/11/2026."
}
```


---

### Authors 
Vedika Shetty (J060)
Aman Singh (J064)
Aaria Srivastav (J066)
Arijit Srivastava (J067)
Adil Syed (J069)
