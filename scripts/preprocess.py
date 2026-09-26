"""Build a traceable NLP dataset from the immutable university PDF corpus.

This module deliberately uses conservative, extractive rules.  It never edits
``data/raw`` and records uncertainty instead of inventing notice information.
All output writing is deterministic (stable source ordering and IDs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - makes the error actionable
    raise SystemExit("pypdf is required. Install project dependencies before running.") from exc


LOG = logging.getLogger("preprocess")
PIPELINE_VERSION = "2.0.0"
MIN_USABLE_TEXT = 40
SIMILARITY_THRESHOLD = 0.95
OCR_MIN_TEXT = 40
SUMMARY_MAX_CHARS = 600


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    output_dir: Path
    english_only: bool = True
    force: bool = False


@dataclass
class SourceDocument:
    document_id: str | None
    pdf_path: Path
    pdf_filename: str
    unverified_filename_id: str | None = None
    university: str | None = None
    title: str | None = None
    category: str | None = None
    metadata_language: str | None = None
    source_url: str | None = None
    metadata_missing: bool = False
    source_metadata: dict[str, str] = field(default_factory=dict)


def clean_value(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def canonical_id(value: str | None) -> str | None:
    """Return the metadata ID as a stable string; do not manufacture IDs."""
    value = clean_value(value)
    if value is None:
        return None
    try:
        return str(int(value))
    except ValueError:
        return value


def id_sort_key(document_id: str | None) -> tuple[int, int | str]:
    if document_id and document_id.isdigit():
        return (0, int(document_id))
    return (1, document_id or "")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)!r}")


def load_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def filename_id(filename: str) -> str | None:
    match = re.match(r"(\d+)", filename)
    return canonical_id(match.group(1)) if match else None


def discover_documents(raw_dir: Path) -> list[SourceDocument]:
    """Join metadata to PDFs by basename, falling back to filename ID only.

    Metadata IDs are authoritative where present.  A PDF absent from metadata is
    still attempted and is explicitly flagged, rather than silently skipped.
    """
    metadata_path = raw_dir / "metadata.csv"
    pdf_dir = raw_dir / "pdfs"
    rows = load_metadata(metadata_path)
    by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        local_path = clean_value(row.get("local_path"))
        if local_path:
            by_filename[Path(local_path).name].append(row)

    documents: list[SourceDocument] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf"), key=lambda item: (id_sort_key(filename_id(item.name)), item.name)):
        matches = by_filename.get(pdf_path.name, [])
        if len(matches) > 1:
            LOG.warning("Multiple metadata rows refer to %s; using the first row", pdf_path.name)
        row = matches[0] if matches else None
        if row:
            documents.append(
                SourceDocument(
                    document_id=canonical_id(row.get("id")),
                    pdf_path=pdf_path,
                    pdf_filename=pdf_path.name,
                    university=clean_value(row.get("institution")),
                    title=clean_value(row.get("subject")),
                    category=clean_value(row.get("category")),
                    metadata_language=clean_value(row.get("language")),
                    source_url=clean_value(row.get("pdf_url")),
                    source_metadata={key: value for key, value in row.items()},
                )
            )
        else:
            # Do not promote a filename-derived number to an authoritative
            # document_id when the metadata row is absent.
            documents.append(
                SourceDocument(
                    document_id=None,
                    pdf_path=pdf_path,
                    pdf_filename=pdf_path.name,
                    unverified_filename_id=filename_id(pdf_path.name),
                    metadata_missing=True,
                )
            )
    return documents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(path: Path) -> dict[str, Any]:
    """Extract text with a layout->default fallback for layout-only failures.

    The raw PDF is never rewritten.  The fallback only changes the extraction
    method used to read the same immutable source.
    """
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return {
                    "raw_text": "",
                    "page_count": 0,
                    "status": "encrypted",
                    "method": None,
                    "error": "PDF is encrypted",
                }

        layout_pages: list[str] = []
        default_pages: list[str] = []
        page_errors: list[str] = []

        for index, page in enumerate(reader.pages, start=1):
            try:
                layout_text = page.extract_text(extraction_mode="layout") or ""
            except Exception as exc:
                layout_text = ""
                page_errors.append(
                    f"page_{index}:layout:{type(exc).__name__}: {exc}"
                )

            # pypdf's default extraction often succeeds when layout extraction
            # returns an empty string for PDFs whose text is stored in an
            # unusual layout structure.
            if len(re.sub(r"\s+", "", layout_text)) >= OCR_MIN_TEXT:
                chosen = layout_text
                default_text = ""
            else:
                try:
                    default_text = page.extract_text() or ""
                except Exception as exc:
                    default_text = ""
                    page_errors.append(
                        f"page_{index}:default:{type(exc).__name__}: {exc}"
                    )
                chosen = (
                    default_text
                    if len(re.sub(r"\s+", "", default_text))
                    > len(re.sub(r"\s+", "", layout_text))
                    else layout_text
                )

            layout_pages.append(layout_text)
            default_pages.append(default_text)
            # Store the selected page text below.
            layout_pages[-1] = chosen

        text = "\n\f\n".join(layout_pages)
        status = "success" if text.strip() else "no_text"

        # Determine whether the selected result came from layout or fallback.
        layout_len = sum(len(re.sub(r"\s+", "", p)) for p in [
            p if p else "" for p in layout_pages
        ])
        default_len = sum(len(re.sub(r"\s+", "", p)) for p in default_pages)
        method = "default_fallback" if default_len > 0 and default_len >= layout_len else "layout"

        return {
            "raw_text": text,
            "page_count": len(reader.pages),
            "status": status,
            "method": method if status == "success" else None,
            "error": "; ".join(page_errors) or None,
        }
    except Exception as exc:
        return {
            "raw_text": "",
            "page_count": 0,
            "status": "failed",
            "method": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def detect_language(text: str) -> str:
    """Script-based language classification that does not need a network model."""
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari >= 20 and latin >= 20:
        return "mixed Hindi/English"
    if devanagari >= 20:
        return "Hindi"
    if latin >= 20:
        return "English"
    return "other/unknown"


def assess_extraction_quality(text: str, page_count: int, status: str) -> tuple[list[str], dict[str, float | int]]:
    """Apply documented, intentionally conservative extraction checks."""
    flags: list[str] = []
    compact = re.sub(r"\s+", "", text)
    characters = len(text)
    replacement_count = text.count("\ufffd") + text.count("ï¿½")
    non_alnum_ratio = (sum(not char.isalnum() and not char.isspace() for char in text) / characters) if characters else 0.0
    if status == "no_text":
        flags.extend(["empty_or_no_text_layer", "requires_ocr"])
    elif status in {"failed", "encrypted"}:
        flags.append("extraction_failed")
    if status == "success" and len(compact) < MIN_USABLE_TEXT:
        flags.append("extremely_short_text")
    if replacement_count >= 3 or (characters and replacement_count / characters > 0.01):
        flags.append("replacement_character_artifacts")
    if characters >= 100 and non_alnum_ratio > 0.35:
        flags.append("high_non_alphanumeric_ratio")
    if "\x00" in text:
        flags.append("control_character_artifacts")
    if page_count > 1 and not text.strip():
        flags.append("suspicious_page_extraction")
    # Consecutive dense columns are a safe warning for tables, not a claim of corruption.
    if sum(1 for line in text.splitlines() if len(re.findall(r"\s{3,}", line)) >= 3) >= 3:
        flags.append("possible_badly_extracted_table")
    return flags, {
        "text_length": characters,
        "non_whitespace_length": len(compact),
        "replacement_character_count": replacement_count,
        "non_alphanumeric_ratio": round(non_alnum_ratio, 4),
    }


def repeated_margin_lines(text: str) -> set[str]:
    pages = text.split("\f")
    candidates: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        for line in lines[:2] + lines[-2:]:
            normalized = re.sub(r"\d+", "#", line.lower())
            if 4 <= len(normalized) <= 160:
                candidates[normalized] += 1
    threshold = 2 if len(pages) >= 2 else 99
    return {line for line, count in candidates.items() if count >= threshold}


def clean_text(raw_text: str) -> tuple[str, list[str]]:
    """Normalize layout artefacts while retaining semantics, case, values and URLs."""
    text = unicodedata.normalize("NFKC", raw_text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char for char in text if char in "\n\t" or unicodedata.category(char)[0] != "C")
    repeated = repeated_margin_lines(text)
    removed_margins = 0
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        normalized = re.sub(r"\d+", "#", stripped.lower())
        if normalized in repeated:
            removed_margins += 1
            continue
        if re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", stripped, re.I):
            continue
        cleaned_lines.append(re.sub(r"[ \t]+", " ", stripped))
    text = "\n".join(cleaned_lines)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)  # a split word only
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z])", " ", text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    flags = ["repeated_header_footer_removed"] if removed_margins else []
    return text, flags


def normalised_for_matching(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def detect_duplicates(records: list[dict[str, Any]]) -> None:
    """Flag duplicate/template similarity without deleting distinct notices.

    Exact normalized-text matches are marked as duplicate candidates.  Near-copy
    similarity is only a review signal: rounds, schedules, forms, and other
    template-reused notices must not be excluded automatically.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        fingerprint = normalised_for_matching(record["cleaned_text"])
        if fingerprint:
            groups[hashlib.sha256(fingerprint.encode()).hexdigest()].append(record)

    for group in groups.values():
        if len(group) > 1:
            group.sort(key=lambda record: id_sort_key(record["document_id"]))
            leader = group[0]
            for duplicate in group[1:]:
                duplicate["duplicate_candidate"] = True
                duplicate["duplicate_of"] = leader["document_id"]
                duplicate["duplicate_reason"] = "exact_normalized_text_match"

    eligible = [record for record in records if record["cleaned_text"]]
    token_sets = {
        id(record): set(re.findall(r"[a-z0-9]{3,}", record["cleaned_text"].lower()))
        for record in eligible
    }
    for index, left in enumerate(eligible):
        left_text = normalised_for_matching(left["cleaned_text"])
        if len(left_text) < 200:
            continue
        for right in eligible[index + 1:]:
            right_text = normalised_for_matching(right["cleaned_text"])
            if len(right_text) < 200:
                continue
            if abs(len(left_text) - len(right_text)) / max(len(left_text), len(right_text)) > 0.08:
                continue
            left_tokens, right_tokens = token_sets[id(left)], token_sets[id(right)]
            union = left_tokens | right_tokens
            score = len(left_tokens & right_tokens) / len(union) if union else 0.0
            if score >= SIMILARITY_THRESHOLD:
                candidate = {
                    "document_id": right["document_id"],
                    "similar_to": left["document_id"],
                    "score": round(score, 4),
                }
                left.setdefault("similarity_candidates", []).append(candidate)
                right.setdefault("similarity_candidates", []).append({
                    "document_id": left["document_id"],
                    "similar_to": right["document_id"],
                    "score": round(score, 4),
                })
                # Deliberately do not set duplicate_candidate here.
                right.setdefault("quality_flags", []).append("near_duplicate_similarity_review")
                left.setdefault("quality_flags", []).append("near_duplicate_similarity_review")


DATE_PATTERNS = [
    re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b"),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)[, ]+\s*(\d{4})\b", re.I),
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s*(\d{4})\b", re.I),
]


DEADLINE_CUES = re.compile(
    r"\b(?:deadline|last\s+date|due\s+date|due|by|before|no\s+later\s+than|"
    r"closing\s+date|closes|submission\s+date|date\s+of\s+submission|"
    r"report(?:ing)?\s+by|register\s+by|apply\s+by|pay\s+by|submit\s+by)\b",
    re.I,
)


def parse_date_match(pattern_index: int, match: re.Match[str]) -> str | None:
    try:
        if pattern_index == 0:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            ).date().isoformat()
        if pattern_index == 1:
            year = int(match.group(3))
            year += 2000 if year < 100 else 0
            return datetime(year, int(match.group(2)), int(match.group(1))).date().isoformat()
        if pattern_index == 2:
            return datetime.strptime(
                " ".join(match.groups()), "%d %B %Y"
            ).date().isoformat()
        return datetime.strptime(
            " ".join((match.group(2), match.group(1), match.group(3))),
            "%d %B %Y",
        ).date().isoformat()
    except ValueError:
        return None


def extract_deadlines(text: str) -> list[str]:
    """Return dates only when local text explicitly supports deadline semantics."""
    deadlines: list[str] = []
    for pattern_index, pattern in enumerate(DATE_PATTERNS):
        for match in pattern.finditer(text):
            start, end = match.span()
            context = text[max(0, start - 100): min(len(text), end + 100)]
            if not DEADLINE_CUES.search(context):
                continue
            value = parse_date_match(pattern_index, match)
            if value and value not in deadlines:
                deadlines.append(value)
    return deadlines


def sentences(text: str) -> list[str]:
    flattened = re.sub(r"\s+", " ", text).strip()
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", flattened)
        if len(item.strip()) >= 20
    ]


ACTION_CUES = re.compile(
    r"\b(?:must|shall|required to|are required to|is required to|"
    r"need to|should|submit|register|apply|pay|upload|attend|report|"
    r"complete|fill|download|verify|confirm|respond)\b",
    re.I,
)


def first_action_sentence(text: str) -> str | None:
    for sentence in sentences(text):
        if ACTION_CUES.search(sentence):
            # Reject common metadata/table/administrative descriptions that are
            # not an instruction to the notice audience.
            low = sentence.lower()
            if any(
                bad in low
                for bad in (
                    "website has been updated",
                    "for information only",
                    "table of contents",
                )
            ):
                continue
            return sentence[:SUMMARY_MAX_CHARS]
    return None


def first_audience_action_sentence(text: str, audience: str) -> str | None:
    patterns = {
        "Students": r"\b(?:student|students|candidate|candidates|scholar|scholars)\b",
        "Faculty": r"\b(?:faculty\s+members?|teaching\s+faculty|dean|head\s+of\s+department|hod)\b",
        "Staff": r"\bstaff\s+members?\b",
        "Researchers": r"\b(?:research\s+scholar|researcher|ph\.?\s*d\.?)\b",
        "Applicants": r"\bapplicants?\b",
        "Alumni": r"\balumni\b",
        "General Public": r"\b(?:general\s+public|all\s+concerned|public\s+notice)\b",
    }
    audience_pattern = patterns[audience]
    for sentence in sentences(text):
        if re.search(audience_pattern, sentence, re.I) and ACTION_CUES.search(sentence):
            return sentence[:SUMMARY_MAX_CHARS]
    return None


AUDIENCE_PATTERNS = {
    "Students": (
        r"\ball\s+students\b",
        r"\bstudents?\s+(?:are|must|shall|should|need\s+to|required|eligible|"
        r"may|can|are\s+requested)\b",
        r"\b(?:eligible\s+)?students?\b(?=.*\b(?:register|apply|submit|attend|pay|upload)\b)",
        r"\bstudents?\s+(?:of|from)\s+(?:the\s+)?(?:university|department|programme|course)\b",
    ),
    "Faculty": (
        r"\bfaculty\s+members?\b",
        r"\bteaching\s+faculty\b",
        r"\ball\s+(?:teaching\s+)?faculty\b",
        r"\bfaculty\s+(?:members?|are|must|shall|should|required)\b",
        r"\b(?:deans?|heads?\s+of\s+department|hods?)\b",
    ),
    "Staff": (
        r"\bstaff\s+members?\b",
        r"\bnon[-\s]?teaching\s+staff\b",
        r"\bstaff\s+(?:are|must|shall|should|required)\b",
    ),
    "Researchers": (
        r"\bresearch\s+scholars?\b",
        r"\bresearchers?\s+(?:are|must|shall|should|required)\b",
        r"\bph\.?\s*d\.?\s+(?:scholars?|researchers?)\b",
    ),
    "Applicants": (
        r"\bapplicants?\s+(?:are|must|shall|should|required|may)\b",
        r"\ball\s+applicants?\b",
        r"\beligible\s+applicants?\b",
        r"\badmission\s+seekers?\b",
    ),
    "Alumni": (
        r"\balumni\s+(?:are|must|shall|should|required|may)\b",
        r"\ball\s+alumni\b",
    ),
    "General Public": (
        r"\bgeneral\s+public\b",
        r"\bpublic\s+notice\b",
        r"\ball\s+concerned\b",
    ),
}


def extract_audience(text: str, title: str | None = None) -> list[str]:
    search_text = f"{title or ''}\n{text}"
    labels: list[str] = []
    for label, patterns in AUDIENCE_PATTERNS.items():
        if any(re.search(pattern, search_text, re.I | re.S) for pattern in patterns):
            labels.append(label)
    return labels


def extract_contact(text: str) -> str | None:
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)
    phones = re.findall(r"(?<!\d)(?:\+91[- ]?)?\d{10}(?!\d)", text)
    values = list(dict.fromkeys(emails + phones))
    return "; ".join(values) if values else None


def infer_category(title: str | None, text: str) -> str | None:
    """Small documented fallback only when authoritative category is absent."""
    content = f"{title or ''} {text[:3000]}".lower()
    rules = (("examination", ("examination", "exam form", "result")), ("admission", ("admission", "admit card", "application form")), ("hostel", ("hostel", "hall", "mess")), ("academic", ("semester", "curriculum", "course")), ("announcements", ("notice", "announcement")))
    for category, terms in rules:
        if any(term in content for term in terms):
            return category
    return None


def build_record(source: SourceDocument, extraction: dict[str, Any]) -> dict[str, Any]:
    raw_text = extraction["raw_text"]
    language = detect_language(raw_text)
    quality_flags, metrics = assess_extraction_quality(
        raw_text, extraction["page_count"], extraction["status"]
    )
    cleaned_text, cleaning_flags = clean_text(raw_text)
    if source.metadata_missing:
        quality_flags.append("metadata_missing")
    category = source.category or infer_category(source.title, cleaned_text)
    if category is None:
        quality_flags.append("category_missing_or_uncertain")

    audience = extract_audience(cleaned_text, source.title)
    action = first_action_sentence(cleaned_text)
    student_summary = (
        first_audience_action_sentence(cleaned_text, "Students")
        if "Students" in audience
        else None
    )
    faculty_summary = (
        first_audience_action_sentence(cleaned_text, "Faculty")
        if "Faculty" in audience
        else None
    )

    document_id = source.document_id
    if document_id and document_id.isdigit():
        notice_id = f"CIRC-2026-{int(document_id):04d}"
    elif source.unverified_filename_id:
        notice_id = f"CIRC-UNVERIFIED-{int(source.unverified_filename_id):04d}"
    else:
        notice_id = None

    return {
        "notice_id": notice_id,
        "document_id": document_id,
        "unverified_filename_id": source.unverified_filename_id,
        "pdf_filename": source.pdf_filename,
        "pdf_path": str(source.pdf_path.as_posix()),
        "pdf_sha256": sha256_file(source.pdf_path),
        "university": source.university,
        "title": source.title,
        "category": category,
        "source_category": source.category,
        "metadata_language": source.metadata_language,
        "language": language,
        "source_url": source.source_url,
        "source_metadata": source.source_metadata,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "page_count": extraction["page_count"],
        "extraction_status": extraction["status"],
        "extraction_method": extraction.get("method"),
        "extraction_error": extraction["error"],
        "extraction_quality": metrics,
        "quality_flags": sorted(set(quality_flags + cleaning_flags)),
        "target_audience": audience,
        "student_summary": student_summary,
        "faculty_summary": faculty_summary,
        "entities": {
            "deadlines": extract_deadlines(cleaned_text),
            "action_required": action,
            "contact": extract_contact(cleaned_text),
        },
        "duplicate_candidate": False,
        "duplicate_of": None,
        "duplicate_reason": None,
        "similarity_candidates": [],
        "processing_status": "needs_review",
        "exclusion_reason": None,
    }


def apply_inclusion_rules(records: list[dict[str, Any]], english_only: bool) -> None:
    for record in records:
        flags = set(record["quality_flags"])
        reason: str | None = None
        status = "included"

        if record["document_id"] is None:
            reason, status = "missing_authoritative_document_id", "needs_review"
        elif record["extraction_status"] != "success":
            reason, status = "unreadable_or_scanned_requires_ocr", "extraction_failed"
        elif record["extraction_quality"]["non_whitespace_length"] < MIN_USABLE_TEXT:
            reason, status = "insufficient_text", "needs_review"
        elif english_only and record["language"] in {"Hindi", "other/unknown"}:
            reason, status = "non_english_detected", "excluded"
        elif english_only and record["language"] == "mixed Hindi/English":
            reason, status = "mixed_language_requires_review", "needs_review"
        elif {"replacement_character_artifacts", "high_non_alphanumeric_ratio"} & flags:
            reason, status = "extraction_quality_requires_review", "needs_review"

        # Exact/near duplicate flags are review metadata only.  A round,
        # schedule, form, or other template-reused notice is still a real
        # notice and must not be silently removed.
        if record["duplicate_candidate"] and status == "included":
            record["quality_flags"].append("exact_duplicate_review")

        if record["similarity_candidates"] and status == "included":
            record["quality_flags"].append("near_duplicate_similarity_review")

        record["processing_status"] = status
        record["exclusion_reason"] = reason


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("pdf_filename", "language", "notice_id")
    for key in required:
        if not record.get(key):
            errors.append(f"missing_{key}")
    if record.get("document_id") is None and not record.get("unverified_filename_id"):
        errors.append("missing_document_id_and_unverified_filename_id")
    if record["extraction_status"] == "success" and not record["raw_text"].strip():
        errors.append("successful_extraction_without_raw_text")
    if record["extraction_status"] == "success" and not record["cleaned_text"].strip():
        errors.append("successful_extraction_without_cleaned_text")
    if not isinstance(record["target_audience"], list):
        errors.append("target_audience_not_list")
    entities = record["entities"]
    if not isinstance(entities, dict) or set(entities) != {"deadlines", "action_required", "contact"}:
        errors.append("invalid_entities_schema")
    elif not isinstance(entities["deadlines"], list) or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) for date in entities["deadlines"]):
        errors.append("invalid_deadline_format")
    if record["processing_status"] in {"excluded", "needs_review", "extraction_failed"} and not record["exclusion_reason"]:
        errors.append("status_missing_exclusion_reason")
    if not Path(record["pdf_path"]).is_file():
        errors.append("source_pdf_missing")
    return errors


def atomic_write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; rerun with --force to regenerate outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], force: bool) -> None:
    atomic_write(path, "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, default=json_default) + "\n" for record in records), force)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; rerun with --force to regenerate outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def quality_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            "document_id": record["document_id"], "pdf_filename": record["pdf_filename"],
            "extraction_status": record["extraction_status"],
            "extraction_method": record["extraction_method"],
            "text_length": record["extraction_quality"]["text_length"], "page_count": record["page_count"], "language": record["language"],
            "quality_flags": json.dumps(record["quality_flags"]), "duplicate_candidate": record["duplicate_candidate"], "duplicate_of": record["duplicate_of"],
            "duplicate_reason": record["duplicate_reason"], "processing_status": record["processing_status"], "exclusion_reason": record["exclusion_reason"],
            "extraction_error": record["extraction_error"],
        })
    return rows


def final_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            "notice_id": record["notice_id"], "document_id": record["document_id"],
            "unverified_filename_id": record["unverified_filename_id"], "university": record["university"], "title": record["title"],
            "category": record["category"], "language": record["language"], "metadata_language": record["metadata_language"], "source_url": record["source_url"],
            "pdf_filename": record["pdf_filename"], "raw_text": record["raw_text"], "cleaned_text": record["cleaned_text"],
            "target_audience": json.dumps(record["target_audience"], ensure_ascii=False), "student_summary": record["student_summary"],
            "faculty_summary": record["faculty_summary"], "deadlines": json.dumps(record["entities"]["deadlines"]),
            "action_required": record["entities"]["action_required"], "contact": record["entities"]["contact"],
        })
    return rows


def generate_summary(records: list[dict[str, Any]], validation_errors: dict[str, list[str]], settings: Settings) -> dict[str, Any]:
    language_counts = Counter(record["language"] for record in records)
    status_counts = Counter(record["processing_status"] for record in records)
    reason_counts = Counter(record["exclusion_reason"] for record in records if record["exclusion_reason"])
    return {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"english_only": settings.english_only, "minimum_usable_non_whitespace_characters": MIN_USABLE_TEXT, "near_duplicate_similarity_threshold": SIMILARITY_THRESHOLD,
            "near_duplicates_are_review_only": True},
        "quality_thresholds": {"replacement_character_flag": "at least 3 characters or more than 1% of text", "high_non_alphanumeric_ratio": "> 0.35 on text of at least 100 characters", "mixed_language": "at least 20 Devanagari and 20 Latin characters"},
        "total_pdfs_found": len(records),
        "successfully_extracted": sum(record["extraction_status"] == "success" for record in records),
        "extraction_failures": sum(record["extraction_status"] in {"failed", "encrypted"} for record in records),
        "empty_or_scanned_documents": sum("empty_or_no_text_layer" in record["quality_flags"] for record in records),
        "language_distribution": dict(sorted(language_counts.items())),
        "exact_duplicate_candidates": sum(record["duplicate_candidate"] for record in records),
        "near_duplicate_review_records": sum(bool(record["similarity_candidates"]) for record in records),
        "processing_status_distribution": dict(sorted(status_counts.items())),
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "included_documents": status_counts["included"],
        "documents_needing_review": status_counts["needs_review"],
        "counts_by_university": dict(sorted(Counter(record["university"] or "unknown" for record in records).items())),
        "counts_by_category": dict(sorted(Counter(record["category"] or "unknown" for record in records).items())),
        "validation_records_with_errors": len(validation_errors),
        "validation_errors": validation_errors,
    }


def save_outputs(records: list[dict[str, Any]], summary: dict[str, Any], settings: Settings) -> None:
    output = settings.output_dir
    extracted = [{
        key: record[key]
        for key in (
            "document_id", "unverified_filename_id", "pdf_filename", "pdf_path",
            "pdf_sha256", "raw_text", "page_count", "extraction_status",
            "extraction_method", "extraction_error", "extraction_quality",
        )
    } for record in records]
    cleaned = [{
        key: record[key]
        for key in (
            "document_id", "unverified_filename_id", "pdf_filename", "raw_text",
            "cleaned_text", "language", "quality_flags",
        )
    } for record in records]
    annotations = [{
        key: record[key]
        for key in (
            "notice_id", "document_id", "unverified_filename_id", "university",
            "title", "category", "language", "metadata_language", "source_url",
            "pdf_filename", "raw_text", "cleaned_text", "target_audience",
            "student_summary", "faculty_summary", "entities", "processing_status",
            "exclusion_reason",
        )
    } for record in records]
    included = [record for record in records if record["processing_status"] == "included"]
    write_jsonl(output / "extracted_text" / "extracted_text.jsonl", extracted, settings.force)
    write_jsonl(output / "cleaned_text" / "cleaned_text.jsonl", cleaned, settings.force)
    write_csv(output / "quality_reports" / "quality_report.csv", quality_rows(records), list(quality_rows(records)[0]) if records else [], settings.force)
    atomic_write(output / "quality_reports" / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", settings.force)
    write_jsonl(output / "annotations" / "annotations.jsonl", annotations, settings.force)
    write_jsonl(output / "final_dataset" / "final_dataset.jsonl", included, settings.force)
    csv_fields = ["notice_id", "document_id", "unverified_filename_id", "university", "title", "category", "language", "metadata_language", "source_url", "pdf_filename", "raw_text", "cleaned_text", "target_audience", "student_summary", "faculty_summary", "deadlines", "action_required", "contact"]
    write_csv(output / "final_dataset" / "final_dataset.csv", final_csv_rows(included), csv_fields, settings.force)
    manifest = [{
        "document_id": record["document_id"],
        "unverified_filename_id": record["unverified_filename_id"],
        "pdf_filename": record["pdf_filename"],
        "pdf_sha256": record["pdf_sha256"],
    } for record in records]
    atomic_write(output / "quality_reports" / "source_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", settings.force)


def run_pipeline(settings: Settings) -> dict[str, Any]:
    LOG.info("Starting preprocessing pipeline version %s", PIPELINE_VERSION)
    documents = discover_documents(settings.raw_dir)
    LOG.info("Discovered %d PDFs", len(documents))
    if not documents:
        raise RuntimeError(f"No PDFs found in {settings.raw_dir / 'pdfs'}")
    records: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        LOG.info("[%d/%d] Processing document_id=%s file=%s", index, len(documents), document.document_id, document.pdf_filename)
        records.append(build_record(document, extract_pdf_text(document.pdf_path)))
    detect_duplicates(records)
    apply_inclusion_rules(records, settings.english_only)
    records.sort(key=lambda record: (id_sort_key(record["document_id"]), record["pdf_filename"]))
    notice_ids = Counter(record["notice_id"] for record in records if record["notice_id"])
    validation_errors = {record["document_id"] or record["pdf_filename"]: validate_record(record) for record in records}
    validation_errors = {key: value for key, value in validation_errors.items() if value}
    if any(count > 1 for count in notice_ids.values()):
        validation_errors["dataset"] = ["notice_id_not_unique"]
    summary = generate_summary(records, validation_errors, settings)
    save_outputs(records, summary, settings)
    LOG.info("Completed: %d included, %d needing review", summary["included_documents"], summary["documents_needing_review"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and validate the university communications PDF dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Immutable raw input directory (default: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Generated-output directory (default: data/processed)")
    parser.add_argument("--include-non-english", action="store_true", help="Make detected Hindi/other languages eligible for final data.")
    parser.add_argument("--force", action="store_true", help="Deterministically replace existing aggregate generated outputs.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = run_pipeline(Settings(raw_dir=args.raw_dir, output_dir=args.output_dir, english_only=not args.include_non_english, force=args.force))
    except (FileExistsError, RuntimeError, OSError) as exc:
        LOG.error("Pipeline stopped: %s", exc)
        return 1
    print(json.dumps({key: summary[key] for key in ("total_pdfs_found", "successfully_extracted", "language_distribution",
         "exact_duplicate_candidates", "near_duplicate_review_records",
         "included_documents", "documents_needing_review")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
