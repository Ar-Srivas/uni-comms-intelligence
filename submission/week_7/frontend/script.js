/**
 * University Communications Intelligence - Frontend Application Logic
 * Pure vanilla JavaScript. Handles tab switching, file validation,
 * sample loading, API integration, and email clipboard copying.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Tabs
  const tabPaste = document.getElementById('tab-paste');
  const tabUpload = document.getElementById('tab-upload');
  const pastePanel = document.getElementById('paste-panel');
  const uploadPanel = document.getElementById('upload-panel');

  // DOM Elements - Inputs
  const noticeTextarea = document.getElementById('notice-text');
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const fileInfo = document.getElementById('file-info');
  const selectedFileName = document.getElementById('selected-file-name');
  const selectedFileSize = document.getElementById('selected-file-size');
  const removeFileBtn = document.getElementById('remove-file-btn');

  // DOM Elements - Action Buttons
  const analyzeBtn = document.getElementById('analyze-btn');
  const loadSampleBtn = document.getElementById('load-sample-btn');
  const clearBtn = document.getElementById('clear-btn');
  const sidebarLoadSampleBtn = document.getElementById('sidebar-load-sample-btn');
  const layoutPreviewBtn = document.getElementById('layout-preview-btn');

  // DOM Elements - Status & Results
  const statusMessage = document.getElementById('status-message');
  const resultsPanel = document.getElementById('results-panel');
  const previewModeTag = document.getElementById('preview-mode-tag');
  const resultCategory = document.getElementById('result-category');
  const resultAudience = document.getElementById('result-audience');
  const extractedEntities = document.getElementById('extracted-entities');
  const studentEmailText = document.getElementById('student-email-text');
  const facultyEmailText = document.getElementById('faculty-email-text');
  const copyStudentBtn = document.getElementById('copy-student-btn');
  const copyFacultyBtn = document.getElementById('copy-faculty-btn');

  // DOM Elements - Configuration
  const apiEndpointInput = document.getElementById('api-endpoint-input');
  const sidebarSamplePreview = document.getElementById('sidebar-sample-preview');

  // State
  let activeTab = 'paste'; // 'paste' | 'upload'
  let uploadedFileContent = null;
  let uploadedFileName = '';
  const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB

  // Sample Notice/Syllabus for Testing
  const SAMPLE_NOTICE = sidebarSamplePreview
    ? sidebarSamplePreview.textContent.trim()
    : `DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
ACADEMIC YEAR 2026-2027 | SEMESTER I

COURSE SYLLABUS UPDATE & EXAMINATION SCHEDULE NOTICE

Course Code: CS-402 (Advanced Machine Learning & NLP)
Instructor: Dr. Eleanor Vance (evance@university.edu)
Office: Turing Hall, Room 304 | Office Hours: Tue/Thu 2:00 PM - 4:00 PM

Dear Students and Faculty Members,

Please take note of the updated course syllabus, revised project submission deadlines, and upcoming midterm examination schedule:

1. Revised Syllabus Topics:
   - Weeks 1-4: Classical Text Classification & Feature Engineering
   - Weeks 5-7: Sequence Labeling & Named Entity Recognition (NER)
   - Weeks 8-11: Sequence-to-Sequence Modeling & Persona-conditioned Summarization
   - Weeks 12-14: Model Evaluation, Robustness Audits, and Production Deployment

2. Key Deadlines:
   - Project Proposal Submission: 15/10/2026 by 11:59 PM via LMS portal
   - Midterm Examination: 05/11/2026 from 10:00 AM to 12:00 PM (Auditorium Hall B)
   - Final Term Project & Code Freeze: 10/12/2026 by 5:00 PM

3. Instructions for Faculty:
   - Teaching assistants must submit rubric scorecards to the Academic Registrar by 20/10/2026.
   - Proctoring assignments will be communicated by Controller of Examinations.

For questions regarding course policies or accommodations, please contact cs-queries@university.edu or visit https://cs.university.edu/courses/cs402.

Approved by:
Prof. Arthur Pendelton
Dean of Academic Affairs
AI University`;

  // Sample Structured Layout Preview (used strictly for offline UI layout validation)
  const SAMPLE_LAYOUT_DATA = {
    category: 'academic',
    audience: ['students', 'faculty'],
    entities: {
      dates: ['15/10/2026', '20/10/2026', '05/11/2026', '10/12/2026'],
      times: ['10:00 AM', '12:00 PM', '2:00 PM', '4:00 PM', '5:00 PM', '11:59 PM'],
      emails: ['evance@university.edu', 'cs-queries@university.edu'],
      urls: ['https://cs.university.edu/courses/cs402'],
      organizations: ['Department of Computer Science & Engineering', 'AI University'],
      persons: ['Dr. Eleanor Vance', 'Prof. Arthur Pendelton'],
      roles: ['Dean of Academic Affairs', 'Instructor', 'Academic Registrar', 'Controller of Examinations']
    },
    summaries: {
      student: `Subject: CS-402 Syllabus Update & Midterm Examination Schedule

Dear CS-402 Students,

Please review the following critical deadlines and updates for CS-402 (Advanced Machine Learning & NLP):

- Project Proposal Due: 15/10/2026 by 11:59 PM via the LMS portal.
- Midterm Examination: 05/11/2026 from 10:00 AM to 12:00 PM in Auditorium Hall B.
- Final Term Project & Code Freeze: 10/12/2026 by 5:00 PM.

The updated course syllabus now covers Classical Classification, Named Entity Recognition, and Sequence-to-Sequence Persona Summarization. Office hours are held Tuesdays & Thursdays from 2:00 PM to 4:00 PM in Turing Hall 304.

For queries or accommodations, email cs-queries@university.edu or check the portal at https://cs.university.edu/courses/cs402.`,

      faculty: `Subject: Faculty Action Items: CS-402 Evaluation Deadlines & Proctoring

Dear CS-402 Teaching Faculty and Assistants,

Please take note of the administrative timeline for CS-402:

1. Rubric Scorecards: All teaching assistants must finalize and submit rubric scorecards to the Academic Registrar by 20/10/2026.
2. Midterm Examination: Scheduled for 05/11/2026 (10:00 AM – 12:00 PM, Auditorium Hall B). Proctoring rosters will be distributed by the Controller of Examinations.
3. Final Project Evaluation: Project code freeze is set for 10/12/2026 at 5:00 PM.

Please contact cs-queries@university.edu for any scheduling discrepancies.

Regards,
Department of Computer Science & Engineering`
    }
  };

  /* ==========================================================================
     Tab Navigation
     ========================================================================== */
  function switchTab(mode) {
    activeTab = mode;
    if (mode === 'paste') {
      tabPaste.classList.add('active');
      tabPaste.setAttribute('aria-selected', 'true');
      tabUpload.classList.remove('active');
      tabUpload.setAttribute('aria-selected', 'false');

      pastePanel.classList.remove('hidden');
      uploadPanel.classList.add('hidden');
      noticeTextarea.focus();
    } else {
      tabUpload.classList.add('active');
      tabUpload.setAttribute('aria-selected', 'true');
      tabPaste.classList.remove('active');
      tabPaste.setAttribute('aria-selected', 'false');

      uploadPanel.classList.remove('hidden');
      pastePanel.classList.add('hidden');
    }
  }

  tabPaste.addEventListener('click', () => switchTab('paste'));
  tabUpload.addEventListener('click', () => switchTab('upload'));

  /* ==========================================================================
     File Handling & Validation
     ========================================================================== */
  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function handleFileSelected(file) {
    clearStatus();

    if (!file) {
      return;
    }

    // Check file extension
    const fileName = file.name || '';
    if (!fileName.toLowerCase().endsWith('.txt')) {
      showStatus('Validation Error: Only plain text (.txt) files are supported.', 'error');
      resetFileInput();
      return;
    }

    // Check file size (5 MB limit)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      showStatus(`Validation Error: File size (${formatBytes(file.size)}) exceeds the 5 MB limit.`, 'error');
      resetFileInput();
      return;
    }

    // Read file content
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedFileContent = e.target.result;
      uploadedFileName = file.name;

      selectedFileName.textContent = file.name;
      selectedFileSize.textContent = `(${formatBytes(file.size)})`;
      fileInfo.classList.remove('hidden');
      showStatus(`File "${file.name}" loaded successfully (${formatBytes(file.size)}). Click "Analyze notice" to process.`, 'info');
    };

    reader.onerror = () => {
      showStatus('Error reading file. Please check file permissions and try again.', 'error');
      resetFileInput();
    };

    reader.readAsText(file, 'UTF-8');
  }

  function resetFileInput() {
    fileInput.value = '';
    uploadedFileContent = null;
    uploadedFileName = '';
    fileInfo.classList.add('hidden');
  }

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  removeFileBtn.addEventListener('click', () => {
    resetFileInput();
    clearStatus();
  });

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  /* ==========================================================================
     Sample Loading & Form Clearing
     ========================================================================== */
  function loadSampleData() {
    noticeTextarea.value = SAMPLE_NOTICE;
    switchTab('paste');
    clearStatus();
    showStatus('Sample syllabus & examination notice loaded into editor.', 'info');
  }

  loadSampleBtn.addEventListener('click', loadSampleData);
  if (sidebarLoadSampleBtn) {
    sidebarLoadSampleBtn.addEventListener('click', loadSampleData);
  }

  clearBtn.addEventListener('click', () => {
    noticeTextarea.value = '';
    resetFileInput();
    clearStatus();
    resultsPanel.classList.add('hidden');
    previewModeTag.classList.add('hidden');
  });

  /* ==========================================================================
     Status Messages
     ========================================================================== */
  function showStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-box ${type}`;
    statusMessage.classList.remove('hidden');
  }

  function clearStatus() {
    statusMessage.textContent = '';
    statusMessage.className = 'status-box hidden';
  }

  /* ==========================================================================
     Rendering Results
     ========================================================================== */
  function renderResults(data, isPreview = false) {
    // 1. Predicted Category
    const categoryText = data.category || (data.primary_category) || 'General';
    resultCategory.textContent = categoryText;

    // 2. Target Audience
    resultAudience.innerHTML = '';
    const audiences = Array.isArray(data.audience)
      ? data.audience
      : (Array.isArray(data.target_audiences) ? data.target_audiences : ['General']);

    if (audiences.length === 0) {
      const span = document.createElement('span');
      span.className = 'badge audience-tag';
      span.textContent = 'All Campus';
      resultAudience.appendChild(span);
    } else {
      audiences.forEach(aud => {
        const span = document.createElement('span');
        span.className = 'badge audience-tag';
        span.textContent = aud;
        resultAudience.appendChild(span);
      });
    }

    // 3. Extracted Information
    extractedEntities.innerHTML = '';
    const entities = data.entities || {};

    const entitySections = [
      { key: 'dates', title: 'Dates & Deadlines' },
      { key: 'times', title: 'Times' },
      { key: 'emails', title: 'Contact Emails' },
      { key: 'urls', title: 'Web Links' },
      { key: 'organizations', title: 'Departments / Orgs' },
      { key: 'persons', title: 'Key Personnel' },
      { key: 'roles', title: 'Institutional Roles' }
    ];

    entitySections.forEach(section => {
      const values = entities[section.key];
      const groupEl = document.createElement('div');
      groupEl.className = 'entity-group';

      const labelEl = document.createElement('div');
      labelEl.className = 'entity-group-label';
      labelEl.textContent = section.title;
      groupEl.appendChild(labelEl);

      if (Array.isArray(values) && values.length > 0) {
        const listEl = document.createElement('ul');
        listEl.className = 'entity-list';
        values.forEach(item => {
          const li = document.createElement('li');
          if (section.key === 'urls') {
            const a = document.createElement('a');
            a.href = item.startsWith('http') ? item : `https://${item}`;
            a.textContent = item;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            li.appendChild(a);
          } else if (section.key === 'emails') {
            const a = document.createElement('a');
            a.href = `mailto:${item}`;
            a.textContent = item;
            li.appendChild(a);
          } else {
            li.textContent = item;
          }
          listEl.appendChild(li);
        });
        groupEl.appendChild(listEl);
      } else {
        const emptyEl = document.createElement('div');
        emptyEl.className = 'entity-empty';
        emptyEl.textContent = 'None detected';
        groupEl.appendChild(emptyEl);
      }

      extractedEntities.appendChild(groupEl);
    });

    // 4. Email Drafts
    const summaries = data.summaries || {};
    const emails = data.emails || {};
    const studentDraft = emails.student || summaries.student || data.student_summary || 'No student-specific action stated.';
    const facultyDraft = emails.faculty || summaries.faculty || data.faculty_summary || 'No faculty-specific action stated.';

    studentEmailText.value = studentDraft;
    facultyEmailText.value = facultyDraft;

    // Toggle Preview Mode Badge
    if (isPreview) {
      previewModeTag.classList.remove('hidden');
    } else {
      previewModeTag.classList.add('hidden');
    }

    // Display Results Panel
    resultsPanel.classList.remove('hidden');
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* ==========================================================================
     Layout Preview Trigger (Layout testing without live model)
     ========================================================================== */
  if (layoutPreviewBtn) {
    layoutPreviewBtn.addEventListener('click', () => {
      renderResults(SAMPLE_LAYOUT_DATA, true);
      showStatus('Displaying layout preview with structured sample data. You can inspect card layout, edit drafts, and test copy buttons.', 'warning');
    });
  }

  /* ==========================================================================
     API Analysis Execution
     ========================================================================== */
  async function analyzeNotice() {
    clearStatus();

    // 1. Gather & Validate Input
    let textToAnalyze = '';
    if (activeTab === 'paste') {
      textToAnalyze = noticeTextarea.value.trim();
    } else {
      textToAnalyze = (uploadedFileContent || '').trim();
    }

    if (!textToAnalyze) {
      const msg = activeTab === 'paste'
        ? 'Validation Error: Please enter or paste notice text before analyzing.'
        : 'Validation Error: Please upload a .txt notice file before analyzing.';
      showStatus(msg, 'error');
      return;
    }

    // 2. Endpoint Configuration
    const endpoint = (apiEndpointInput && apiEndpointInput.value.trim())
      ? apiEndpointInput.value.trim()
      : '/process-notice';

    // 3. UI Loading State
    setLoading(true);
    showStatus('Processing notice through NLP pipeline (Stage 1 Classification, Stage 2 NER, Stage 3 Summarization)...', 'processing');

    const payload = {
      doc_id: `notice_${Date.now()}`,
      text: textToAnalyze
    };

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.detail) {
            errorDetail = typeof errJson.detail === 'string'
              ? errJson.detail
              : JSON.stringify(errJson.detail);
          }
        } catch (_) {
          // If response is not JSON, retain default errorDetail
        }
        throw new Error(errorDetail);
      }

      const resultData = await response.json();
      renderResults(resultData, false);
      showStatus('Notice analyzed successfully. Results and audience-specific drafts generated.', 'success');

    } catch (err) {
      console.error('NLP Backend Analysis Error:', err);

      let helpfulMessage = '';
      if (err.name === 'TypeError' || err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        helpfulMessage = `Unable to connect to NLP backend at "${endpoint}". Please ensure the server is running. You can also click "Preview UI layout with sample data" to test UI interactions.`;
      } else {
        helpfulMessage = `Analysis failed: ${err.message}`;
      }

      showStatus(helpfulMessage, 'error');
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    const spinner = analyzeBtn.querySelector('.btn-spinner');
    const text = analyzeBtn.querySelector('.btn-text');

    if (isLoading) {
      analyzeBtn.disabled = true;
      if (spinner) spinner.classList.remove('hidden');
      if (text) text.textContent = 'Analyzing...';
    } else {
      analyzeBtn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
      if (text) text.textContent = 'Analyze notice';
    }
  }

  analyzeBtn.addEventListener('click', analyzeNotice);

  /* ==========================================================================
     Clipboard Copy Functionality
     ========================================================================== */
  function copyEmailDraft(button, targetId) {
    const textarea = document.getElementById(targetId);
    if (!textarea) return;

    const textToCopy = textarea.value;
    if (!textToCopy.trim()) {
      showStatus('Nothing to copy: the email draft is empty.', 'warning');
      return;
    }

    const originalText = button.textContent;

    const showCopiedFeedback = () => {
      button.textContent = 'Copied!';
      button.classList.add('copied');
      setTimeout(() => {
        button.textContent = originalText;
        button.classList.remove('copied');
      }, 2000);
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(textToCopy)
        .then(showCopiedFeedback)
        .catch(err => {
          console.warn('Clipboard writeText failed, attempting execCommand fallback:', err);
          fallbackCopy(textarea, showCopiedFeedback);
        });
    } else {
      fallbackCopy(textarea, showCopiedFeedback);
    }
  }

  function fallbackCopy(textarea, callback) {
    textarea.focus();
    textarea.select();
    try {
      const successful = document.execCommand('copy');
      if (successful) {
        callback();
      } else {
        showStatus('Clipboard copy command was not successful. Please manually copy the text.', 'warning');
      }
    } catch (err) {
      console.error('execCommand copy error:', err);
      showStatus('Unable to copy automatically. Please select and copy the text manually.', 'error');
    }
  }

  copyStudentBtn.addEventListener('click', () => copyEmailDraft(copyStudentBtn, 'student-email-text'));
  copyFacultyBtn.addEventListener('click', () => copyEmailDraft(copyFacultyBtn, 'faculty-email-text'));
});