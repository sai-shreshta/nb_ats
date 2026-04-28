# ATS — Resume Screener

An internal HR tool that screens and ranks multiple resumes against a job description using AI (Google Gemini or Anthropic Claude). No authentication required.

---

## Quick Start

### 1. Install dependencies

```bash
cd ats
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
```

Open `.env` and fill in your API key(s):

```env
# Choose provider: "gemini" or "claude"
LLM_PROVIDER=gemini

# Gemini settings
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash          # change to any valid Gemini model

# Anthropic / Claude settings
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CLAUDE_MODEL=claude-sonnet-4-5         # change to any valid Claude model
```

> **Only the key for your chosen provider is required.**
> You can leave the other key blank or use a placeholder.

### 3. Run the server

```bash
uvicorn main:app --reload
```

The app will be available at **http://127.0.0.1:8000**

---

## Switching Between Gemini and Claude

Edit `.env` and change `LLM_PROVIDER`:

| Provider | Value |
|---|---|
| Google Gemini | `LLM_PROVIDER=gemini` |
| Anthropic Claude | `LLM_PROVIDER=claude` |

You can also change the exact model within each provider:

```env
GEMINI_MODEL=gemini-2.5-flash      # or gemini-1.5-pro, etc.
CLAUDE_MODEL=claude-sonnet-4-5     # or claude-3-5-haiku-20241022, etc.
```

Restart the server after any `.env` change.

---

## Usage

1. Open **http://127.0.0.1:8000** in your browser
2. Paste the **Job Description** into the left panel
3. **Upload** one or more resumes (`.pdf` or `.docx`) — drag and drop supported
4. Click **Screen Resumes**
5. View ranked results in the right panel:
   - **Shortlisted** tab: candidates who scored ≥ 70
   - **All Candidates** tab: every resume, ranked by score
6. Click **Export CSV** to download all results

---

## File Structure

```
ats/
├── main.py          # FastAPI app — /api/screen endpoint
├── llm_client.py    # Gemini + Claude abstraction
├── parser.py        # PDF + DOCX text extraction
├── requirements.txt
├── .env.example     # Environment variable template
└── static/
    └── index.html   # Single-page frontend
README.md
```

---

## API

**`POST /api/screen`**

| Field | Type | Description |
|---|---|---|
| `jd` | `string` (form) | Full job description text |
| `files` | `file[]` (form) | One or more `.pdf` / `.docx` resume files |

**Response:** JSON array sorted by `overall_score` descending.

```json
[
  {
    "candidate_name": "Jane Doe",
    "filename": "jane_doe_cv.pdf",
    "overall_score": 87,
    "skills_score": 36,
    "experience_score": 27,
    "education_score": 16,
    "presentation_score": 8,
    "shortlisted": true,
    "strengths": ["5 years Python", "FastAPI experience", "Strong ML background"],
    "gaps": ["No Kubernetes experience"],
    "summary": "Strong overall fit for the role..."
  }
]
```

---

## Requirements

- Python 3.9+
- Internet access (for LLM API calls)
- A valid Gemini or Claude API key
