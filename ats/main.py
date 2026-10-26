"""
main.py — FastAPI ATS application, Vercel-compatible.
"""
import asyncio
import os
import uuid
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

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "sai.shreshta@nobroker.in").strip().lower()

RESUME_BUCKET = "resumes"
# The bucket is private; CVs are served through signed links. They end up in
# CSV exports that get shared with recruiters, so they must outlive a session.
CV_LINK_TTL = 7 * 24 * 60 * 60  # 7 days

# Only these columns are accepted from the client when saving candidates.
CANDIDATE_FIELDS = (
    "candidate_name", "phone_number", "email", "city", "resume_filename", "resume_url",
    "overall_score", "skills_score", "experience_score", "education_score",
    "presentation_score", "shortlisted", "strengths", "gaps", "summary",
)

app = FastAPI(title="ATS — Resume Screener", version="5.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _supabase():
    """Service-role client. Never sign a user in on this client: supabase-py
    swaps the client's Authorization header to the user's JWT on sign-in,
    after which queries run under RLS as that user."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
    return create_client(url, key)


def _auth_client():
    """Throwaway client used only for password sign-in and token refresh."""
    return _supabase()


def _get_profile(sb, user_id: str, columns: str) -> dict | None:
    result = sb.table("profiles").select(columns).eq("id", user_id).limit(1).execute()
    return result.data[0] if result.data else None


def _select_all(query_fn, page_size: int = 1000) -> list:
    """PostgREST caps responses at 1000 rows; page through everything.
    query_fn() must return a fresh, ordered query builder each call."""
    rows: list = []
    start = 0
    while True:
        batch = query_fn().range(start, start + page_size - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        start += page_size


def _get_current_user(authorization: str | None):
    """Validate Bearer token, return user dict. Raises HTTPException on failure."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header.")
    token = authorization.split(" ", 1)[1]
    sb = _supabase()
    try:
        resp = sb.auth.get_user(token)
        user = resp.user
    except Exception:
        user = None
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"id": str(user.id), "email": (user.email or "").lower()}


def _require_approved(authorization: str | None):
    """Validate token and check user is approved. Returns user dict."""
    user = _get_current_user(authorization)
    # Admin email always has access
    if user["email"] == ADMIN_EMAIL:
        user["role"] = "admin"
        return user
    profile = _get_profile(_supabase(), user["id"], "approved, role")
    if not profile:
        raise HTTPException(status_code=403, detail="Account not found. Please request access.")
    if not profile.get("approved"):
        raise HTTPException(status_code=403, detail="Your account is pending admin approval.")
    user["role"] = profile.get("role") or "user"
    return user


def _require_admin(authorization: str | None):
    """Validate token and check user is admin."""
    user = _require_approved(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ── Storage helpers ───────────────────────────────────────────────────────────

def _storage_path(value: str | None) -> str:
    """Normalise a stored resume reference to its object path in the bucket.
    Accepts a bare path, an old public URL, or a signed URL."""
    if not value:
        return ""
    value = value.split("?", 1)[0]
    marker = f"/{RESUME_BUCKET}/"
    if "://" in value:
        return value.split(marker, 1)[1] if marker in value else ""
    return value.lstrip("/")


def _sign_paths(sb, paths: list[str], expires_in: int) -> dict[str, str]:
    """Return {path: signed_url} for the given object paths."""
    paths = sorted({p for p in paths if p})
    if not paths:
        return {}
    signed: dict[str, str] = {}
    try:
        for i in range(0, len(paths), 500):
            items = sb.storage.from_(RESUME_BUCKET).create_signed_urls(paths[i:i + 500], expires_in)
            for item in items:
                url = item.get("signedURL") or item.get("signedUrl")
                if url and item.get("path"):
                    signed[item["path"]] = url
    except Exception:
        pass
    return signed


def _attach_signed_urls(sb, rows: list[dict], expires_in: int) -> list[dict]:
    """Replace each row's resume_url with a fresh signed URL (or '')."""
    paths = {id(r): _storage_path(r.get("resume_url")) for r in rows}
    signed = _sign_paths(sb, list(paths.values()), expires_in)
    for r in rows:
        r["resume_url"] = signed.get(paths[id(r)], "")
    return rows


def _upload_resume_to_storage(sb, file_bytes: bytes, filename: str) -> str:
    """Upload resume bytes to Supabase Storage. Returns the object path or '' on failure."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if ext not in content_types:
        return ""
    try:
        storage_path = f"{uuid.uuid4()}.{ext}"
        sb.storage.from_(RESUME_BUCKET).upload(
            storage_path,
            file_bytes,
            {"content-type": content_types[ext]},
        )
        return storage_path
    except Exception:
        return ""


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
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

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
    is_admin = email == ADMIN_EMAIL
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
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password are required.")

    try:
        resp = _auth_client().auth.sign_in_with_password({"email": email, "password": password})
        session = resp.session
        user = resp.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not session or not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    sb = _supabase()
    user_id = str(user.id)
    profile = _get_profile(sb, user_id, "approved, role, full_name")

    if email == ADMIN_EMAIL:
        # Ensure admin profile exists
        if not profile:
            sb.table("profiles").insert({
                "id": user_id,
                "email": email,
                "full_name": "Admin",
                "approved": True,
                "role": "admin",
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "approved_by": "system",
            }).execute()
            full_name = "Admin"
        else:
            full_name = profile.get("full_name") or "Admin"
        role = "admin"
    else:
        if not profile:
            raise HTTPException(status_code=403, detail="Account not found. Please request access.")
        if not profile.get("approved"):
            return JSONResponse({"status": "pending"})
        role = profile.get("role") or "user"
        full_name = profile.get("full_name") or ""

    return JSONResponse({
        "status": "ok",
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "email": user.email,
        "full_name": full_name,
        "role": role,
    })


@app.post("/api/auth/refresh")
async def refresh(request: Request) -> JSONResponse:
    body = await request.json()
    refresh_token = body.get("refresh_token") or ""
    if not refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token is required.")
    try:
        resp = _auth_client().auth.refresh_session(refresh_token)
        session = resp.session
    except Exception:
        session = None
    if not session:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    return JSONResponse({
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    })


@app.get("/api/auth/me")
async def me(authorization: str | None = Header(default=None)) -> JSONResponse:
    user = _get_current_user(authorization)
    profile = _get_profile(_supabase(), user["id"], "full_name, role, approved")
    if user["email"] == ADMIN_EMAIL:
        return JSONResponse({
            "email": user["email"],
            "full_name": (profile or {}).get("full_name") or "Admin",
            "role": "admin",
            "approved": True,
        })
    if not profile:
        return JSONResponse({"email": user["email"], "role": "user", "approved": False})
    return JSONResponse({
        "email": user["email"],
        "full_name": profile.get("full_name") or "",
        "role": profile.get("role") or "user",
        "approved": bool(profile.get("approved")),
    })


# ── Core ATS endpoints (require approved user) ────────────────────────────────

@app.post("/api/extract")
async def extract_resumes(
    files: List[UploadFile] = File(...),
    store: bool = Form(True),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    sb = _supabase()

    async def _one(upload: UploadFile) -> dict:
        filename = upload.filename or "unknown_file"
        file_bytes = await upload.read()
        resume_path = ""
        if store:
            resume_path = await asyncio.to_thread(_upload_resume_to_storage, sb, file_bytes, filename)
        result = {"filename": filename, "resume_path": resume_path}
        try:
            result["resume_text"] = await asyncio.to_thread(extract_text, file_bytes, filename)
        except ValueError as e:
            result["resume_text"] = None
            result["error"] = str(e)
        return result

    results = list(await asyncio.gather(*[_one(f) for f in files]))
    signed = _sign_paths(sb, [r["resume_path"] for r in results], CV_LINK_TTL)
    for r in results:
        r["resume_url"] = signed.get(r["resume_path"], "")
    return JSONResponse(results)


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
    raw_candidates = body.get("candidates") or []
    sb = _supabase()
    result = sb.table("screenings").insert({
        "job_role": body.get("job_role") or "Unknown Role",
        "jd_text": body.get("jd_text") or "",
        "total_screened": int(body.get("total_screened") or len(raw_candidates)),
        "total_shortlisted": int(body.get("total_shortlisted") or 0),
        "recorded_by": user["email"],
    }).execute()
    screening_id = result.data[0]["id"]
    candidates = []
    for c in raw_candidates:
        row = {k: c.get(k) for k in CANDIDATE_FIELDS if k in c}
        # Store the bucket path, never a (signed) URL.
        row["resume_url"] = _storage_path(c.get("resume_path") or c.get("resume_url"))
        row["screening_id"] = screening_id
        candidates.append(row)
    if candidates:
        sb.table("candidates").insert(candidates).execute()
    return JSONResponse({"id": screening_id})


@app.get("/api/history")
async def get_history(
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    sb = _supabase()
    screenings = _select_all(
        lambda: sb.table("screenings")
        .select("id, created_at, job_role, total_screened, total_shortlisted, recorded_by")
        .order("created_at", desc=True)
    )
    return JSONResponse({"screenings": screenings})


@app.get("/api/history/{screening_id}/candidates")
async def get_screening_candidates(
    screening_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_approved(authorization)
    sb = _supabase()
    rows = _select_all(
        lambda: sb.table("candidates").select("*").eq("screening_id", screening_id)
        .order("overall_score", desc=True).order("id")
    )
    return JSONResponse(_attach_signed_urls(sb, rows, CV_LINK_TTL))


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
    admin = _require_admin(authorization)
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="You cannot revoke your own access.")
    sb = _supabase()
    sb.table("profiles").update({"approved": False, "approved_at": None, "approved_by": None}).eq("id", user_id).execute()
    return JSONResponse({"ok": True})


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
    user_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    admin = _require_admin(authorization)
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    sb = _supabase()
    # Delete auth user (cascades to profile)
    try:
        sb.auth.admin.delete_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not delete user: {e}")
    return JSONResponse({"ok": True})


@app.put("/api/admin/users/{user_id}/role")
async def admin_set_role(
    user_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    admin = _require_admin(authorization)
    body = await request.json()
    role = body.get("role")
    if role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'.")
    if user_id == admin["id"] and role != "admin":
        raise HTTPException(status_code=400, detail="You cannot remove your own admin role.")
    sb = _supabase()
    sb.table("profiles").update({"role": role}).eq("id", user_id).execute()
    return JSONResponse({"ok": True})


@app.delete("/api/history/{screening_id}")
async def delete_screening(
    screening_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    sb = _supabase()
    rows = _select_all(
        lambda: sb.table("candidates").select("id, resume_url").eq("screening_id", screening_id).order("id")
    )
    paths = [p for p in (_storage_path(r.get("resume_url")) for r in rows) if p]
    sb.table("candidates").delete().eq("screening_id", screening_id).execute()
    sb.table("screenings").delete().eq("id", screening_id).execute()
    # Best effort: remove the stored CVs too.
    for i in range(0, len(paths), 100):
        try:
            sb.storage.from_(RESUME_BUCKET).remove(paths[i:i + 100])
        except Exception:
            pass
    return JSONResponse({"ok": True})


@app.get("/api/tracker")
async def get_tracker(
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_admin(authorization)
    sb = _supabase()

    screenings = _select_all(
        lambda: sb.table("screenings")
        .select("id, job_role, total_screened, total_shortlisted, recorded_by, created_at")
        .order("id")
    )
    candidates = _select_all(
        lambda: sb.table("candidates").select("phone_number, email, overall_score, shortlisted").order("id")
    )

    profiles_res = sb.table("profiles").select("email, full_name").execute()
    name_map = {(p.get("email") or "").lower(): p.get("full_name") or "" for p in (profiles_res.data or [])}

    total_runs = len(screenings)
    total_screened = sum(s.get("total_screened") or 0 for s in screenings)
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

    avg_score = round(sum(c.get("overall_score") or 0 for c in candidates) / max(total_applications, 1), 1)
    shortlist_rate = round(total_shortlisted / max(total_applications, 1) * 100, 1)

    user_stats: dict = {}
    for s in screenings:
        key = s.get("recorded_by") or "unknown"
        if key not in user_stats:
            user_stats[key] = {
                "recorded_by": key,
                "full_name": name_map.get(key.lower(), ""),
                "runs": 0, "screened": 0, "shortlisted": 0, "last_run": None,
            }
        user_stats[key]["runs"] += 1
        user_stats[key]["screened"] += s.get("total_screened") or 0
        user_stats[key]["shortlisted"] += s.get("total_shortlisted") or 0
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
        role_stats[role]["screened"] += s.get("total_screened") or 0
        role_stats[role]["shortlisted"] += s.get("total_shortlisted") or 0

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
    role_map: dict = {}
    rows: list = []
    # Chunk the id list to keep the request URL short.
    for i in range(0, len(screening_ids), 100):
        ids = screening_ids[i:i + 100]
        screenings_res = sb.table("screenings").select("id, job_role").in_("id", ids).execute()
        role_map.update({s["id"]: s["job_role"] for s in (screenings_res.data or [])})
        rows.extend(_select_all(
            lambda: sb.table("candidates").select("*").in_("screening_id", ids).order("id")
        ))

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

    return JSONResponse(_attach_signed_urls(sb, list(seen.values()), CV_LINK_TTL))


# Serve static files — local dev only. On Vercel all traffic hits the function.
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
