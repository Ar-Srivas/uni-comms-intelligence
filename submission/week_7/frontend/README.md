# University Communications Intelligence - Frontend

A dedicated, lightweight, and responsive web frontend for the **University Communications Intelligence** NLP platform. Built entirely with plain HTML5, CSS3, and vanilla JavaScript (no React, Tailwind, Bootstrap, or build dependencies).

---

## 1. Design & Aesthetic Philosophy

The interface follows a traditional, professional university dashboard aesthetic:
- **Clean Academic Styling:** Crisp white background (`#ffffff`), subtle grey panels (`#f6f8fa`), thin borders (`#d0d7de`), and restrained academic navy accents (`#0d233a`, `#12385f`).
- **Typography:** Georgia-style serif headings for an authoritative, institutional appearance combined with clean, highly readable system sans-serif body text and monospace code/input areas.
- **No Extraneous Modern AI Gimmicks:** Zero gradients, no neon glows, no glassmorphism, no oversized rounded cards, and no distracting decorative illustrations.
- **Responsive Layout:** 
  - **Desktop:** Main content column (65-70%) on the left paired with a contextual institutional sidebar (30-35%) on the right. Side-by-side student and faculty email cards.
  - **Mobile:** Fluid single-column stacked layout with full usability down to narrow screen widths (360px).

---

## 2. File Structure

```
frontend/
├── index.html     # Semantic HTML5 document with accessible elements, tabs, and panels
├── styles.css     # Modular vanilla CSS containing design tokens, academic styles, and responsive media queries
├── script.js      # Vanilla JavaScript handling tab navigation, file validation, API integration, and clipboard copy
└── README.md      # Documentation, setup instructions, and API contract details
```

---

## 3. How to Run Locally

The FastAPI server serves both the frontend and API. From `submission/week_7`:

```bash
uvicorn src.api:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

---

## 4. Backend API Integration & Contract

The frontend connects to the multi-stage NLP pipeline served by `src/api.py`.

### Endpoint Configuration
- **Default Endpoint:** `/process-notice` (same server)
- **Method:** `POST`
- **Configurable in UI:** The sidebar includes an **API Endpoint** input field where you can point the frontend to any port or remote host without modifying code.

### Request Schema
```json
{
  "doc_id": "notice_1698000000000",
  "text": "Dear Students and Faculty, The tuition fee deadline has been revised to 25/11/2026..."
}
```

### Response Schema
```json
{
  "doc_id": "notice_1698000000000",
  "title": "Course Syllabus Update & Examination Notice",
  "category": "academic",
  "audience": ["students", "faculty"],
  "entities": {
    "dates": ["25/11/2026", "05/12/2026"],
    "times": ["10:00 AM", "5:00 PM"],
    "emails": ["exams@university.edu"],
    "urls": ["https://portal.university.edu"],
    "organizations": ["Department of Computer Science"],
    "persons": ["Dr. Eleanor Vance"],
    "roles": ["Dean of Academic Affairs", "Registrar"]
  },
  "summaries": {
    "student": "Dear Students, please take note of the revised deadline of 25/11/2026...",
    "faculty": "Dear Faculty, invigilation and grading rosters are due...",
    "admin": "Administrative notice details..."
  }
}
```

---

## 5. Offline Layout Preview & Testing

When the heavy transformer models (`summarizer_final` weights) or FastAPI daemon are not currently running, the frontend remains **100% testable**:
- **"Load sample" Button:** Populates the text area with a complete realistic university syllabus update and examination notice.
- **"Preview UI layout with sample data" Button:** Located in the sidebar. Renders structured sample metadata, entity extraction grids, and editable student/faculty email drafts.
- A prominent yellow **[Layout Preview Mode]** tag is displayed when previewing to clearly distinguish test data from real backend model predictions.
- **Copy Email:** Click **"Copy email"** on either card to test clipboard copying, accompanied by a visual "Copied!" feedback indicator.
- **Editable Drafts:** Users can modify the text directly inside each card before copying.
- **File Validation Testing:** Supports `.txt` uploads and rejects non-text files or files exceeding the 5 MB limit.
