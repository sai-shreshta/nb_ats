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
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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


# Serve static files — local dev only. On Vercel all traffic hits the function.
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
