"""
main.py — FastAPI ATS application, Vercel-compatible.

Two stateless endpoints — no database, no queue, no shared state:
  POST /api/extract  — Extract text from a batch of resume files.
                       Returns [{filename, resume_text}] to the browser.
  POST /api/score    — Score one resume against a JD, return result.

All job state (extracted texts, scores, progress) lives in the browser.
The frontend uploads files in batches of 10, then drives scoring in parallel
waves of 5 with a cooldown — each function call stays well under 10 s.
"""
import asyncio
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from supabase import create_client

from llm_client import score_resume
from parser import extract_text

load_dotenv()

app = FastAPI(title="ATS — Resume Screener", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/extract")
async def extract_resumes(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Extract plain text from a batch of PDF/DOCX files.
    Returns a list of {filename, resume_text} or {filename, error} objects.
    Used for both JD files and resume files.
    """
    async def _one(upload: UploadFile) -> dict:
        filename = upload.filename or "unknown_file"
        file_bytes = await upload.read()
        try:
            text = await asyncio.to_thread(extract_text, file_bytes, filename)
            return {"filename": filename, "resume_text": text}
        except ValueError as e:
            return {"filename": filename, "resume_text": None, "error": str(e)}

    results = await asyncio.gather(*[_one(f) for f in files])
    return JSONResponse(list(results))


@app.post("/api/score")
async def score_one(
    jd: str = Form(...),
    filename: str = Form(...),
    resume_text: str = Form(...),
) -> JSONResponse:
    """Score one resume against the job description and return the result."""
    result = await score_resume(jd, resume_text, filename)
    return JSONResponse(result)


def _supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
    return create_client(url, key)


@app.post("/api/save-screening")
async def save_screening(request: Request) -> JSONResponse:
    body = await request.json()
    sb = _supabase()
    result = sb.table("screenings").insert({
        "job_role": body["job_role"],
        "jd_text": body["jd_text"],
        "total_screened": body["total_screened"],
        "total_shortlisted": body["total_shortlisted"],
    }).execute()
    screening_id = result.data[0]["id"]
    candidates = [{**c, "screening_id": screening_id} for c in body["candidates"]]
    sb.table("candidates").insert(candidates).execute()
    return JSONResponse({"id": screening_id})


@app.get("/api/history")
async def get_history() -> JSONResponse:
    sb = _supabase()
    screenings = sb.table("screenings").select("*").order("created_at", desc=True).execute()
    candidates = sb.table("candidates").select("phone_number, email, overall_score, shortlisted").execute()
    return JSONResponse({"screenings": screenings.data, "candidates": candidates.data})


@app.get("/api/history/{screening_id}/candidates")
async def get_screening_candidates(screening_id: str) -> JSONResponse:
    sb = _supabase()
    result = sb.table("candidates").select("*").eq("screening_id", screening_id).order("overall_score", desc=True).execute()
    return JSONResponse(result.data)


# Serve static files — local dev only. On Vercel all traffic hits the function.
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
