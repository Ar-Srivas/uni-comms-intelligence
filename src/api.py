from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib

from src.emails import build_emails
from src.integrated_pipeline import (
    IntegratedNoticePipeline,
    NoticeClassifier,
    NoticeEntityExtractor,
    PersonaSummarizer,
    clean_ocr_text,
    strip_table_noise,
)

pipeline_instance: IntegratedNoticePipeline | None = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path(__file__).resolve().parent / "models"
SUMMARIZER_DIR = MODELS_ROOT / "summarizer_final"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


class NoticeRequest(BaseModel):
    doc_id: str = Field(..., example="notice_2026_001")
    text: str = Field(..., example="Dear Students, Tuition fee payment deadline is 20.07.2025. Late fee Rs. 500.")


class NoticeEntities(BaseModel):
    dates: list[str]
    times: list[str]
    emails: list[str]
    urls: list[str]
    organizations: list[str]
    persons: list[str]
    roles: list[str]


class NoticeResponse(BaseModel):
    doc_id: str
    title: str
    category: str
    audience: list[str]
    entities: NoticeEntities
    summaries: dict[str, str]
    emails: dict[str, str] = Field(default_factory=dict)


def load_pipeline() -> None:
    global pipeline_instance
    model_files = {
        "cat_vec.joblib": MODELS_ROOT / "cat_vec.joblib",
        "cat_model.joblib": MODELS_ROOT / "cat_model.joblib",
        "aud_vec.joblib": MODELS_ROOT / "aud_vec.joblib",
        "aud_model.joblib": MODELS_ROOT / "aud_model.joblib",
        "mlb.joblib": MODELS_ROOT / "mlb.joblib",
        "summarizer_final/config.json": SUMMARIZER_DIR / "config.json",
        "summarizer_final/model.safetensors": SUMMARIZER_DIR / "model.safetensors",
        "summarizer_final/tokenizer.json": SUMMARIZER_DIR / "tokenizer.json",
        "summarizer_final/tokenizer_config.json": SUMMARIZER_DIR / "tokenizer_config.json",
    }
    missing = [
        f"src/models/{name}"
        for name, path in model_files.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Required model artifacts are missing: {', '.join(missing)}"
        )

    classifier = NoticeClassifier(
        cat_vec=joblib.load(MODELS_ROOT / "cat_vec.joblib"),
        cat_model=joblib.load(MODELS_ROOT / "cat_model.joblib"),
        aud_vec=joblib.load(MODELS_ROOT / "aud_vec.joblib"),
        aud_model=joblib.load(MODELS_ROOT / "aud_model.joblib"),
        mlb=joblib.load(MODELS_ROOT / "mlb.joblib"),
        cat_repeat=2,
        aud_repeat=2,
    )

    extractor = NoticeEntityExtractor(model_name="dslim/bert-base-NER")
    summarizer = PersonaSummarizer(model_dir=str(SUMMARIZER_DIR))
    pipeline_instance = IntegratedNoticePipeline(classifier, extractor, summarizer)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    load_pipeline()
    yield


app = FastAPI(title="Uni-comms-tool", version="1.0.0", lifespan=lifespan)


@app.get('/health')
def health_check():
    return {"status": "running"}

@app.post("/process-notice", response_model=NoticeResponse)
def process_notice(payload: NoticeRequest):
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="NLP Pipeline is not initialized.")

    try:
        result = pipeline_instance.process_document(doc_id=payload.doc_id, raw_text=payload.text)
        result["emails"] = build_emails(result, payload.text, clean_ocr_text, strip_table_noise)
        return result
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(err)}")


app.mount(
    "/",
    StaticFiles(directory=FRONTEND_ROOT, html=True),
    name="frontend",
)