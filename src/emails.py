from __future__ import annotations

import re

DATED_LINE_RE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{0,4}",
    re.I,
)
FACULTY_LINE_RE = re.compile(
    r"\b(faculty|teaching assistants?|TAs?|instructors?|proctor\w*|invigilat\w*|rubric\w*|grading|registrar|controller|coordinators?|question papers?)\b",
    re.I,
)
# Rule-based safety net: the audience model is skewed towards students, so if the
# notice clearly addresses faculty, we still produce a faculty email from the text.
FACULTY_CUE_RE = re.compile(
    r"\b(instructions? for faculty|faculty members?|course (?:instructors?|coordinators?)|teaching assistants?)\b", re.I
)
AUDIENCE_LABEL = {"student": "Students", "faculty": "Faculty Members", "admin": "Administrative Staff"}


def _clean_bullet(line: str) -> str:
    return re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()


def _dated_lines(text: str, limit: int = 8) -> list[str]:
    out = []
    for line in (text or "").split("\n"):
        l = _clean_bullet(line)
        if 8 < len(l) < 220 and DATED_LINE_RE.search(l) and not re.search(r"academic year", l, re.I):
            if l not in out:
                out.append(l)
    return out[:limit]


def _faculty_lines(text: str, limit: int = 6) -> list[str]:
    out = []
    for line in (text or "").split("\n"):
        l = _clean_bullet(line)
        if 12 < len(l) < 220 and FACULTY_LINE_RE.search(l) and not re.match(r"^(?:instructions? for|dear|subject|sub:|re:)", l, re.I):
            if l not in out:
                out.append(l)
    return out[:limit]


def build_formal_email(persona: str, title: str, category: str, summary: str,
                       source_text: str, entities: dict) -> str:
    """Fixed formal layout; the model summary fills only the 'Summary' slot."""
    topic = re.sub(r"^\s*(?:subject|sub|re)\s*[:\-\u2013]\s*", "", title or "", flags=re.I).strip() or "University Notice"
    audience = AUDIENCE_LABEL.get(persona, "All")
    cat = (category or "general").replace("_", " ")

    purpose = {
        "student": f'I am writing to inform you of an important {cat} update regarding "{topic}". Please review the details below and note any actions required from you.',
        "faculty": f'I am writing to share an official {cat} update regarding "{topic}" and to outline the actions expected from faculty.',
        "admin": f'This is to notify you of an official {cat} update regarding "{topic}" that may require administrative follow-up.',
    }.get(persona, f'Please find below an official update regarding "{topic}".')

    lines = [f"Subject: {topic}", "", f"Dear {audience},", "", purpose, ""]

    if summary and not summary.lower().startswith("no specific"):
        lines += ["Summary", summary.strip(), ""]

    dated = _dated_lines(source_text)
    # Keep each audience's own deadlines: students don't get faculty-only lines and vice versa.
    if persona == "student":
        dated = [d for d in dated if not FACULTY_LINE_RE.search(d)] or dated
    elif persona == "faculty":
        mine = [d for d in dated if FACULTY_LINE_RE.search(d)]
        dated = mine or dated
    if dated:
        lines += ["Key Dates and Deadlines"] + [f"  \u2022 {d}" for d in dated] + [""]

    if persona == "faculty":
        fac = _faculty_lines(source_text)
        if fac:
            lines += ["Action Required from Faculty"] + [f"  \u2022 {a}" for a in fac] + [""]
    elif persona == "student":
        lines += ["Action Required",
                  "  \u2022 Please note the dates above and complete all submissions through the official channels before the deadlines.",
                  ""]

    contacts = [f"  \u2022 Email: {e}" for e in entities.get("emails", [])] + \
               [f"  \u2022 Web: {u}" for u in entities.get("urls", [])]
    if contacts:
        lines += ["Contact and Further Information"] + contacts + [""]

    roles = entities.get("roles", [])
    signer = roles[0] if roles else "Office of Academic Affairs"
    lines += ["Thank you for your attention to this matter.", "", "Regards,", signer]
    return "\n".join(lines)


def build_emails(result: dict, raw_text: str, clean_fn, strip_fn) -> dict:
    """Build one formal email per persona from a pipeline result dict.

    If the model skipped faculty but the notice clearly addresses faculty, a
    rule-based faculty email is built from the source text (no model summary).
    """
    src_clean, _ = strip_fn(clean_fn(raw_text))
    emails = {}
    for persona, summary in result.get("summaries", {}).items():
        skipped = summary.lower().startswith("no specific")
        rule_based = False
        if skipped and persona == "faculty" and FACULTY_CUE_RE.search(src_clean):
            summary, skipped, rule_based = "", False, True
        if skipped:
            emails[persona] = summary
        else:
            emails[persona] = build_formal_email(
                persona, result.get("title", ""), result.get("category", ""),
                summary, src_clean, result.get("entities", {}),
            )
            if rule_based:
                result["summaries"][persona] = "(Rule-based: faculty audience detected from notice text.)"
    return emails