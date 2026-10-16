from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from pathlib import Path
from src.integrated_pipeline import (
    IntegratedNoticePipeline,
    NoticeClassifier,
    NoticeEntityExtractor,
    PersonaSummarizer,
)

app = FastAPI(title="Uni-comms-tool",version="1.0.0",)

pipeline_instance: IntegratedNoticePipeline | None = None
WEEK6_ROOT = Path(__file__).resolve().parent.parent


class NoticeRequest(BaseModel):
    doc_id: str = Field(..., example="notice_2026_001")
    text: str = Field(..., example="Dear Students, Tuition fee payment deadline is 20.07.2025. Late fee Rs. 500.")


class NoticeResponse(BaseModel):
    doc_id: str
    title: str
    category: str
    audience: list[str]
    entities: dict
    summaries: dict[str, str]


@app.on_event("startup")
def load_pipeline():
    global pipeline_instance
    models_dir = WEEK6_ROOT / "models"
    required_artifacts = (
        "cat_vec.joblib",
        "cat_model.joblib",
        "aud_vec.joblib",
        "aud_model.joblib",
        "mlb.joblib",
    )
    missing_artifacts = [name for name in required_artifacts if not (models_dir / name).is_file()]
    summarizer_dir = models_dir / "summarizer_final"
    if missing_artifacts or not (summarizer_dir / "config.json").is_file():
        missing = missing_artifacts or ["summarizer_final/config.json"]
        raise FileNotFoundError(
            f"Week 6 model artifacts are missing from {models_dir}: {', '.join(missing)}"
        )

    # 1. Load Stage 1 Classifier artifacts (joblib)
    cat_vec = joblib.load(models_dir / "cat_vec.joblib")
    cat_model = joblib.load(models_dir / "cat_model.joblib")
    aud_vec = joblib.load(models_dir / "aud_vec.joblib")
    aud_model = joblib.load(models_dir / "aud_model.joblib")
    mlb = joblib.load(models_dir / "mlb.joblib")

    classifier = NoticeClassifier(
        cat_vec=cat_vec,
        cat_model=cat_model,
        aud_vec=aud_vec,
        aud_model=aud_model,
        mlb=mlb,
        cat_repeat=2,
        aud_repeat=2,
    )

    # 2. Load Stage 2 Transformer Entity Extractor
    extractor = NoticeEntityExtractor(model_name="dslim/bert-base-NER")

    # 3. Load Stage 3 Fine-Tuned BART Summarizer
    summarizer = PersonaSummarizer(model_dir=str(summarizer_dir))

    # Instantiate integrated asset
    pipeline_instance = IntegratedNoticePipeline(classifier, extractor, summarizer)


@app.get('/health')
def health_check():
    return {"status": "running"}

@app.post("/process-notice", response_model=NoticeResponse)
def process_notice(payload: NoticeRequest):
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="NLP Pipeline is not initialized.")

    try:
        result = pipeline_instance.process_document(doc_id=payload.doc_id, raw_text=payload.text)
        return result
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(err)}")