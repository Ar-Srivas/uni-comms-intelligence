"""Integrated University Notice Pipeline (Group 11, Week 6 Package)."""
from __future__ import annotations

import re
import json
import difflib
from pathlib import Path
import torch
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Shared Regex Patterns
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
NER_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b")

def clean_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[^\S\n]+", " ", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1-\2", t)
    return t.strip()

def flatten_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def extract_title(text: str, max_chars: int = 200) -> str:
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if not lines:
        return ""
    for line in lines[:10]:
        if re.match(r"^\s*(?:subject|sub|re)\s*[:\-–]\s*", line, re.I):
            return line[:max_chars]
    for line in lines[:5]:
        if len(line.split()) >= 4:
            return line[:max_chars]
    return lines[0][:max_chars]

def build_stage1_text(text: str, title: str | None, repeat: int) -> str:
    if repeat <= 0 or not title:
        return text
    return " ".join([title] * repeat) + " " + text

def strip_table_noise(text: str, digit_ratio: float = 0.7, min_len: int = 10):
    kept, removed = [], 0
    for line in (text or "").split("\n"):
        compact = re.sub(r"\s+", "", line)
        if compact and len(compact) > min_len and sum(c.isdigit() for c in compact) / len(compact) > digit_ratio:
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed

ROLE_CANONICAL = {
    "vice chancellor": "Vice-Chancellor", "vc": "Vice-Chancellor",
    "registrar": "Registrar", "deputy registrar": "Deputy Registrar",
    "controller of examinations": "Controller of Examinations",
    "dean": "Dean", "director": "Director", "principal": "Principal"
}

def extract_roles(text: str):
    clean = text.lower()
    found = []
    for key, canonical in ROLE_CANONICAL.items():
        if re.search(r"\b" + re.escape(key) + r"\b", clean):
            found.append(canonical)
    return list(set(found))

# Masking & Resolution Helpers for Stage 3 Model
def mask_entities(text: str, urls: list, emails: list) -> str:
    masked = text
    for index, url in sorted(enumerate(urls), key=lambda pair: -len(pair[1])):
        masked = masked.replace(url, f"[URL{index + 1}]")

    for index, email in enumerate(emails):
        masked = re.sub(re.escape(email), f"[EMAIL{index + 1}]", masked, flags=re.IGNORECASE)

    masked = re.sub(r"https?://[^\s<>\":')\]]+", "[LINK]", masked)
    return masked

def resolve_placeholders(summary: str, urls: list, emails: list) -> str:
    def replace_url(match):
        index = int(match.group(1)) - 1
        return urls[index] if 0 <= index < len(urls) else ""

    def replace_email(match):
        index = int(match.group(1)) - 1
        return emails[index] if 0 <= index < len(emails) else ""

    summary = re.sub(r"\[URL(\d+)\]", replace_url, summary)
    summary = re.sub(r"\[EMAIL(\d+)\]", replace_email, summary)
    summary = summary.replace("[LINK]", "")
    return flatten_whitespace(summary)

def build_model_input(role: str, category: str, dates: list, notice_text: str) -> str:
    date_part = "; ".join(dates[:5]) if dates else "none"
    return f"summarize for {role} | category: {category} | dates: {date_part} | notice: {notice_text}"

def decide_personas(audience: list[str]) -> dict:
    s = "students" in audience or "student" in audience
    f = "faculty" in audience
    a = "admin" in audience or "administrator" in audience
    
    # Default to all active if no specific audience was predicted
    if not s and not f and not a:
        return {"student": "generate", "faculty": "generate", "admin": "generate"}
    
    return {
        "student": "generate" if s else "null",
        "faculty": "generate" if f else "null",
        "admin": "generate" if a else "null",
    }


# --- MODEL 1: STAGE 1 CLASSIFIER ---
class NoticeClassifier:
    def __init__(self, cat_vec, cat_model, aud_vec, aud_model, mlb, cat_repeat=0, aud_repeat=0, thresholds=None):
        self.cat_vec = cat_vec
        self.cat_model = cat_model
        self.aud_vec = aud_vec
        self.aud_model = aud_model
        self.mlb = mlb
        self.cat_repeat = cat_repeat
        self.aud_repeat = aud_repeat
        self.thresholds = thresholds or {c: 0.5 for c in mlb.classes_}

    def predict(self, text: str, title: str | None = None) -> dict:
        t = extract_title(text) if title is None else title
        c_text = build_stage1_text(text, t, self.cat_repeat)
        a_text = build_stage1_text(text, t, self.aud_repeat)

        c_prob = self.cat_model.predict_proba(self.cat_vec.transform([c_text]))[0]
        a_prob = self.aud_model.predict_proba(self.aud_vec.transform([a_text]))[0]

        category = str(self.cat_model.classes_[c_prob.argmax()])
        audience = [cls for cls, p in zip(self.mlb.classes_, a_prob) if p >= self.thresholds.get(cls, 0.5)]

        return {"title": t, "category": category, "audience": audience}


# --- MODEL 2: STAGE 2 NER EXTRACTOR ---
class NoticeEntityExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER", device: int = -1):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = hf_pipeline("ner", model=self.model_name, aggregation_strategy="simple", device=self.device)
        return self._pipe

    def extract(self, text: str) -> dict:
        flat = flatten_whitespace(text)

        dates = list(set(NER_DATE_RE.findall(flat)))
        times = list(set(TIME_RE.findall(flat)))
        emails = list(set(EMAIL_RE.findall(flat)))
        urls = list(set(URL_RE.findall(flat)))
        roles = extract_roles(flat)

        orgs, persons = [], []
        if flat.strip():
            try:
                ner_results = self.pipe(flat[:1200])
                for item in ner_results:
                    group = item.get("entity_group") or item.get("entity") or ""
                    word = item.get("word", "").strip()
                    if group == "ORG" and len(word) > 2:
                        orgs.append(word)
                    elif group == "PER" and len(word) > 2:
                        persons.append(word)
            except Exception:
                pass

        return {
            "dates": dates, "times": times, "emails": emails, "urls": urls,
            "organizations": list(set(orgs)), "persons": list(set(persons)), "roles": roles,
        }


# --- MODEL 3: FINE-TUNED persona SUMMARIZER ---
class PersonaSummarizer:
    def __init__(self, model_dir: str, device: str | None = None):
        self.model_dir = str(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device).eval()

    def generate_summary(self, text: str, persona: str, category: str, dates: list[str], urls: list[str], emails: list[str]) -> str:
        self._load()
        masked_text = flatten_whitespace(mask_entities(text, urls, emails))
        prompt = build_model_input(persona, category, dates, masked_text)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=120,
                min_new_tokens=15,
                num_beams=4,
                no_repeat_ngram_size=6,
                early_stopping=True,
                generation_config=None,
            )
        
        raw_summary = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return resolve_placeholders(raw_summary, urls, emails)


# --- CONSOLIDATED PIPELINE ---
class IntegratedNoticePipeline:
    def __init__(self, classifier: NoticeClassifier, extractor: NoticeEntityExtractor, summarizer: PersonaSummarizer):
        self.classifier = classifier
        self.extractor = extractor
        self.summarizer = summarizer

    def process_document(self, doc_id: str, raw_text: str) -> dict:
        text = clean_ocr_text(raw_text)

        s1 = self.classifier.predict(text)
        s2 = self.extractor.extract(text)
        plan = decide_personas(s1["audience"])
        src_clean, _ = strip_table_noise(text)
        
        summaries = {}
        for p in ("student", "faculty", "admin"):
            if plan[p] == "null":
                summaries[p] = f"No specific {p} action stated."
            else:
                summaries[p] = self.summarizer.generate_summary(
                    src_clean,
                    persona=p,
                    category=s1["category"],
                    dates=s2["dates"],
                    urls=s2["urls"],
                    emails=s2["emails"],
                )

        return {
            "doc_id": str(doc_id),
            "title": s1["title"],
            "category": s1["category"],
            "audience": s1["audience"],
            "entities": s2,
            "summaries": summaries
        }