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

You can run the frontend using any standard local HTTP server or by opening `index.html` directly in a browser.

### Option A: Python Built-In HTTP Server (Recommended)
From the project root directory:

```bash
python3 -m http.server 3000 --directory frontend
```

Then open your browser to:
[http://localhost:3000](http://localhost:3000)

### Option B: Node.js `npx serve` or `http-server`
```bash
npx serve frontend -p 3000
```

### Option C: Direct File Access
Double-click `frontend/index.html` or open it in Google Chrome, Firefox, or Safari:
```bash
open frontend/index.html
```

---

## 4. Backend API Integration & Contract

The frontend connects directly to the multi-stage NLP pipeline exposed in `submission/week_6/src/api.py`.

### Endpoint Configuration
- **Default Endpoint:** `http://localhost:8000/process-notice`
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

### Running the Backend Server
To start the FastAPI backend from the repository root:

```bash
uvicorn submission.week_6.src.api:app --reload --host 0.0.0.0 --port 8000
```

> **Note on CORS:** If serving the frontend from a different port (e.g. `localhost:3000`) and calling the backend at `localhost:8000`, ensure FastAPI includes `CORSMiddleware` in `api.py`:
> ```python
> from fastapi.middleware.cors import CORSMiddleware
> 
> app.add_middleware(
>     CORSMiddleware,
>     allow_origins=["*"],
>     allow_credentials=True,
>     allow_methods=["*"],
>     allow_headers=["*"],
> )
> ```

---

## 5. Offline Layout Preview & Testing

When the heavy transformer models (`summarizer_final` weights) or FastAPI daemon are not currently running, the frontend remains **100% testable**:
- **"Load sample" Button:** Populates the text area with a complete realistic university syllabus update and examination notice.
- **"Preview UI layout with sample data" Button:** Located in the sidebar. Renders structured sample metadata, entity extraction grids, and editable student/faculty email drafts.
- A prominent yellow **[Layout Preview Mode]** tag is displayed when previewing to clearly distinguish test data from real backend model predictions.
- **Copy Email:** Click **"Copy email"** on either card to test clipboard copying, accompanied by a visual "Copied!" feedback indicator.
- **Editable Drafts:** Users can modify the text directly inside each card before copying.
- **File Validation Testing:** Supports `.txt` uploads and rejects non-text files or files exceeding the 5 MB limit.
