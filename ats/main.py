"""
main.py — FastAPI ATS application.

Endpoints:
  POST /api/screen        — Accept JD + resume files, start a background screening job.
                            Returns {job_id, total} immediately.
  GET  /api/jobs/{job_id} — Poll job progress and results.
  GET  /                  — Serves the frontend (index.html via StaticFiles).
"""
import asyncio
import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from llm_client import score_resume
from parser import extract_text

load_dotenv()

app = FastAPI(title="ATS — Resume Screener", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. Fine for single-server local/deployed use.
# For multi-instance deployment, replace with Redis.
_jobs: Dict[str, dict] = {}


def _make_job(total: int) -> dict:
    return {
        "status": "running",
        "total": total,
        "completed": 0,
        "results": [],
        "error": None,
    }


def _parse_error_result(filename: str, message: str) -> dict:
    return {
        "candidate_name": filename.rsplit(".", 1)[0] if "." in filename else filename,
        "filename": filename,
        "overall_score": 0,
        "skills_score": 0,
        "experience_score": 0,
        "education_score": 0,
        "presentation_score": 0,
        "shortlisted": False,
        "strengths": [],
        "gaps": [],
        "summary": "",
        "error": message,
    }


async def _run_job(job_id: str, jd_text: str, resume_items: List[tuple]) -> None:
    """
    Process all resumes concurrently. Updates job state as each result comes in.
    PDF/DOCX parsing runs in a thread pool; LLM calls are async with a semaphore
    (see llm_client.py) so we never exceed the API rate limit.
    """
    job = _jobs[job_id]

    async def process_one(filename: str, file_bytes: bytes) -> None:
        try:
            resume_text = await asyncio.to_thread(extract_text, file_bytes, filename)
        except ValueError as e:
            result = _parse_error_result(filename, f"Could not extract text: {e}")
        else:
            result = await score_resume(jd_text, resume_text, filename)

        job["results"].append(result)
        job["completed"] += 1

    try:
        tasks = [process_one(fn, fb) for fn, fb in resume_items]
        await asyncio.gather(*tasks, return_exceptions=True)
        job["results"].sort(key=lambda c: c.get("overall_score", 0), reverse=True)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/screen")
async def screen_resumes(
    background_tasks: BackgroundTasks,
    jd: str = Form("", description="Job description text"),
    jd_file: Optional[UploadFile] = File(None, description="Job description as PDF or DOCX"),
    files: List[UploadFile] = File(..., description="Resume files (.pdf or .docx)"),
) -> JSONResponse:
    """
    Start a background screening job. Returns {job_id, total} immediately.
    Poll GET /api/jobs/{job_id} for progress and results.
    """
    jd_text = jd.strip()
    if jd_file and jd_file.filename:
        try:
            jd_bytes = await jd_file.read()
            jd_text = extract_text(jd_bytes, jd_file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Could not read JD file: {e}")

    if not jd_text:
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty. Provide text or upload a PDF/DOCX.",
        )
    if not files:
        raise HTTPException(status_code=400, detail="At least one resume file is required.")

    # Read all file bytes now — UploadFile objects are not usable after the response is sent.
    resume_items: List[tuple] = []
    for upload in files:
        filename = upload.filename or "unknown_file"
        file_bytes = await upload.read()
        resume_items.append((filename, file_bytes))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = _make_job(len(resume_items))
    background_tasks.add_task(_run_job, job_id, jd_text, resume_items)

    return JSONResponse({"job_id": job_id, "total": len(resume_items)})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    """Return current job status, progress counts, and accumulated results."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(job)


# Serve static files (frontend) — mount AFTER API routes.
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
