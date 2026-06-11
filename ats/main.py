"""
main.py — FastAPI ATS application, Vercel-compatible.
"""
# redeploy
import asyncio
import os
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from supabase import create_client

from llm_client import score_resume
from parser import extract_text

load_dotenv()

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "sai.shreshta@nobroker.in")

app = FastAPI(title="ATS — Resume Screener", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
    return create_client(url, key)


def _get_current_user(authorization: str | None):
    """Validate Bearer token, return user dict. Raises HTTPException on failure."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")
    token = authorization.split(" ", 1)[1]
    sb = _supabase()
    try:
        resp = sb.auth.get_user(token)
        user = resp.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"id": str(user.id), "email": user.email}


def _require_approved(authorization: str | None):
    """Validate token and check user is approved. Returns user dict."""
    user = _get_current_user(authorization)
    # Admin email always has access
    if user["email"] == ADMIN_EMAIL:
        user["role"] = "admin"
        return user
    sb = _supabase()
    result = sb.table("profiles").select("approved, role").eq("id", user["id"]).single().execute()
    if not result.data:
        raise HTTPException(status_code=403, detail="Account not found. Please request access.")
    if not result.data.get("approved"):
        raise HTTPException(status_code=403, detail="Your account is pending admin approval.")
    user["role"] = result.data.get("role", "user")
    return user


def _require_admin(authorization: str | None):
    """Validate token and check user is admin."""
    user = _require_approved(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/signup")
async def signup(request: Request) -> JSONResponse:
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    full_name = (body.get("full_name") or "").strip()

    if not email or not password or not full_name:
        raise HTTPException(status_code=400, detail="email, password, and full_name are required.")
    if not email.endswith("@nobroker.in"):
        raise HTTPException(status_code=400, detail="Only @nobroker.in email addresses are allowed.")

    sb = _supabase()

    # Check if profile already exists (re-request)
    existing = sb.table("profiles").select("id, approved").eq("email", email).execute()
    if existing.data:
        if existing.data[0].get("approved"):
            return JSONResponse({"status": "already_approved"})
        return JSONResponse({"status": "pending"})

    # Create auth user
    try:
        auth_resp = sb.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True,  # skip email confirmation for internal tool
        })
        user_id = str(auth_resp.user.id)
    except Exception as e:
        msg = str(e)
        if "already registered" in msg.lower() or "already been registered" in msg.lower():
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
        raise HTTPException(status_code=400, detail=msg)

    # Determine if this is the admin email
    is_admin = email == ADMIN_EMAIL.lower()
    sb.table("profiles").insert({
        "id": user_id,
        "email": email,
        "full_name": full_name,
        "approved": is_admin,
        "role": "admin" if is_admin else "user",
        "approved_at": datetime.now(timezone.utc).isoformat() if is_admin else None,
        "approved_by": "system" if is_admin else None,
    }).execute()

    return JSONResponse({"status": "admin" if is_admin else "pending"})


@app.post("/api/auth/login")
async def login(request: Request) -> JSONResponse:
    body = await request.json()
    email = (body.get("email") or "").strip()
    password = body.get("password") or ""

    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password are required.")

    sb = _supabase()
    try:
        resp = sb.auth.sign_in_with_password({"email": email, "password": password})
        session = resp.session
        user = resp.user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Check approval
    is_admin = email.lower() == ADMIN_EMAIL.lower()
    if not is_admin:
        profile = sb.table("profiles").select("approved, role, full_name").eq("id", str(user.id)).single().execute()
        if not profile.data:
            raise HTTPException(status_code=403, detail="Account not found. Please request access.")
        if not profile.data.get("approved"):
            return JSONResponse({"status": "pending"})
        role = profile.data.get("role", "user")
        full_name = profile.data.get("full_name", "")
    else:
        # Ensure admin profile exists
        existing = sb.table("profiles").select("id, role, full_name").eq("id", str(user.id)).execute()
        if not existing.data:
            sb.table("profiles").insert({
                "id": str(user.id),
                "email": email.lower(),
                "full_name": "Admin",
                "approved": True,
                "role": "admin",
            }).execute()
            full_name = "Admin"
        else:
            full_name = existing.data[0].get("full_name", "Admin")
        role = "admin"

    return JSONResponse({
        "status": "ok",
        "access_token": session.access_token,
        "email": user.email,
        "full_name": full_name,
        "role": role,
    })


@app.get("/api/auth/me")
async def me(authorization: str | None = Header(default=None)) -> JSONResponse:
    user = _get_current_user(authorization)
    sb = _supabase()
    profile = sb.table("profiles").select("full_name, role, approved").eq("id", user["id"]).single().execute()
    if not profile.data:
        return JSONResponse({"email": user["email"], "role": "user", "approved": False})
    return JSONResponse({
        "email": user["email"],
        "full_name": profile.data.get("full_name", ""),
        "role": profile.data.get("role", "user"),
        "approved": profile.data.get("approved", False),
    })


# ── Core ATS endpoints (require approved user) ────────────────────────────────

@app.post("/api/extract")
async def extract_resumes(
    files: List[UploadFile] = File(...),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)

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
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    result = await score_resume(jd, resume_text, filename)
    return JSONResponse(result)


@app.post("/api/save-screening")
async def save_screening(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    user = _require_approved(authorization)
    body = await request.json()
    sb = _supabase()
    result = sb.table("screenings").insert({
        "job_role": body["job_role"],
        "jd_text": body["jd_text"],
        "total_screened": body["total_screened"],
        "total_shortlisted": body["total_shortlisted"],
        "recorded_by": user["email"],
    }).execute()
    screening_id = result.data[0]["id"]
    candidates = [{**c, "screening_id": screening_id} for c in body["candidates"]]
    sb.table("candidates").insert(candidates).execute()
    return JSONResponse({"id": screening_id})


@app.get("/api/history")
async def get_history(
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    sb = _supabase()
    screenings = sb.table("screenings").select("*").order("created_at", desc=True).execute()
    candidates = sb.table("candidates").select("phone_number, email, overall_score, shortlisted").execute()
    return JSONResponse({"screenings": screenings.data, "candidates": candidates.data})


@app.get("/api/history/{screening_id}/candidates")
async def get_screening_candidates(
    screening_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    sb = _supabase()
    result = sb.table("candidates").select("*").eq("screening_id", screening_id).order("overall_score", desc=True).execute()
    return JSONResponse(result.data)


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.get("/api/admin/users")
async def admin_list_users(
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    sb = _supabase()
    result = sb.table("profiles").select("*").order("requested_at", desc=True).execute()
    return JSONResponse(result.data)


@app.post("/api/admin/users/{user_id}/approve")
async def admin_approve_user(
    user_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    admin = _require_admin(authorization)
    sb = _supabase()
    sb.table("profiles").update({
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": admin["email"],
    }).eq("id", user_id).execute()
    return JSONResponse({"ok": True})


@app.post("/api/admin/users/{user_id}/revoke")
async def admin_revoke_user(
    user_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    sb = _supabase()
    sb.table("profiles").update({"approved": False, "approved_at": None, "approved_by": None}).eq("id", user_id).execute()
    return JSONResponse({"ok": True})


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
    user_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    sb = _supabase()
    # Delete auth user (cascades to profile)
    sb.auth.admin.delete_user(user_id)
    return JSONResponse({"ok": True})


@app.put("/api/admin/users/{user_id}/role")
async def admin_set_role(
    user_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    body = await request.json()
    role = body.get("role")
    if role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")
    sb = _supabase()
    sb.table("profiles").update({"role": role}).eq("id", user_id).execute()
    return JSONResponse({"ok": True})


@app.get("/api/tracker")
async def get_tracker(
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    sb = _supabase()

    screenings_res = sb.table("screenings").select("id, job_role, total_screened, total_shortlisted, recorded_by, created_at").execute()
    screenings = screenings_res.data or []

    candidates_res = sb.table("candidates").select("phone_number, email, overall_score, shortlisted").execute()
    candidates = candidates_res.data or []

    profiles_res = sb.table("profiles").select("email, full_name").execute()
    name_map = {p["email"]: p.get("full_name", "") for p in (profiles_res.data or [])}

    total_runs = len(screenings)
    total_screened = sum(s.get("total_screened", 0) for s in screenings)
    total_applications = len(candidates)
    total_shortlisted = sum(1 for c in candidates if c.get("shortlisted"))

    seen_phones: set = set()
    seen_emails: set = set()
    unique_count = 0
    for c in candidates:
        phone = (c.get("phone_number") or "").strip()
        email = (c.get("email") or "").strip().lower()
        if phone:
            if phone not in seen_phones:
                seen_phones.add(phone)
                unique_count += 1
        elif email:
            if email not in seen_emails:
                seen_emails.add(email)
                unique_count += 1
        else:
            unique_count += 1

    avg_score = round(sum(c.get("overall_score", 0) for c in candidates) / max(total_applications, 1), 1)
    shortlist_rate = round(total_shortlisted / max(total_applications, 1) * 100, 1)

    user_stats: dict = {}
    for s in screenings:
        key = s.get("recorded_by") or "unknown"
        if key not in user_stats:
            user_stats[key] = {
                "recorded_by": key,
                "full_name": name_map.get(key, ""),
                "runs": 0, "screened": 0, "shortlisted": 0, "last_run": None,
            }
        user_stats[key]["runs"] += 1
        user_stats[key]["screened"] += s.get("total_screened", 0)
        user_stats[key]["shortlisted"] += s.get("total_shortlisted", 0)
        run_date = s.get("created_at")
        if run_date and (not user_stats[key]["last_run"] or run_date > user_stats[key]["last_run"]):
            user_stats[key]["last_run"] = run_date

    by_user = sorted(user_stats.values(), key=lambda x: x["runs"], reverse=True)
    for u in by_user:
        u["shortlist_rate"] = round(u["shortlisted"] / max(u["screened"], 1) * 100, 1)

    role_stats: dict = {}
    for s in screenings:
        role = s.get("job_role") or "Unknown"
        if role not in role_stats:
            role_stats[role] = {"job_role": role, "runs": 0, "screened": 0, "shortlisted": 0}
        role_stats[role]["runs"] += 1
        role_stats[role]["screened"] += s.get("total_screened", 0)
        role_stats[role]["shortlisted"] += s.get("total_shortlisted", 0)

    by_role = sorted(role_stats.values(), key=lambda x: x["runs"], reverse=True)
    for r in by_role:
        r["shortlist_rate"] = round(r["shortlisted"] / max(r["screened"], 1) * 100, 1)

    return JSONResponse({
        "totals": {
            "runs": total_runs,
            "screened": total_screened,
            "applications": total_applications,
            "unique": unique_count,
            "shortlisted": total_shortlisted,
            "avg_score": avg_score,
            "shortlist_rate": shortlist_rate,
        },
        "by_user": by_user,
        "by_role": by_role,
    })


@app.post("/api/export/candidates")
async def export_candidates(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    body = await request.json()
    screening_ids = body.get("screening_ids") or []
    if not screening_ids:
        raise HTTPException(status_code=400, detail="screening_ids is required.")

    sb = _supabase()
    # Fetch screenings to get job_role labels
    screenings_res = sb.table("screenings").select("id, job_role").in_("id", screening_ids).execute()
    role_map = {s["id"]: s["job_role"] for s in (screenings_res.data or [])}

    # Fetch all candidates for those screenings
    candidates_res = sb.table("candidates").select("*").in_("screening_id", screening_ids).execute()
    rows = candidates_res.data or []

    # Attach job_role and deduplicate by phone → email → treat as unique
    # When duplicate, keep the row with the highest overall_score
    seen: dict[str, dict] = {}
    for row in rows:
        row["applied_role"] = role_map.get(row.get("screening_id"), "")
        key = (row.get("phone_number") or "").strip()
        if not key:
            key = (row.get("email") or "").strip().lower()
        if not key:
            key = f"__unique_{row['id']}"
        existing = seen.get(key)
        if existing is None or (row.get("overall_score") or 0) > (existing.get("overall_score") or 0):
            seen[key] = row

    return JSONResponse(list(seen.values()))


# Serve static files — local dev only. On Vercel all traffic hits the function.
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
