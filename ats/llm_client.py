"""
llm_client.py — Async LLM client supporting Gemini and Claude.

Reads configuration from environment variables:
  LLM_PROVIDER      : "gemini" | "claude"
  GEMINI_API_KEY    : Gemini API key
  GEMINI_MODEL      : Gemini model name (default: gemini-3.1-flash-lite-preview)
  ANTHROPIC_API_KEY : Anthropic API key
  CLAUDE_MODEL      : Claude model name (default: claude-sonnet-4-6)
  LLM_CONCURRENCY   : Max simultaneous LLM calls (default: 10)
"""
import asyncio
import json
import os
import random
import re

from dotenv import load_dotenv

load_dotenv()

# Semaphore caps concurrent LLM calls to stay within API rate limits.
# Increase LLM_CONCURRENCY if you have a high-RPM paid tier.
_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "10"))
_semaphore = asyncio.Semaphore(_CONCURRENCY)

PROMPT_TEMPLATE = """You are an expert recruiter evaluating a candidate's resume against a job description.

Before scoring, carefully read the JD and identify:
1. The core nature of the work (e.g. phone-based calling, field sales, software engineering, data analysis, etc.)
2. The must-have skills and experience — things explicitly required, not just mentioned
3. The nice-to-haves — preferred but not essential

Then evaluate the resume with these principles:
- Match the candidate's ACTUAL day-to-day work against what the role requires, not just job titles or industry names. A "Sales Manager" in a field sales role is very different from a "Sales Manager" in an inside sales role — look at what they actually did.
- Be skeptical of keyword overlap. A resume mentioning the same industry or function as the JD is not automatically a good match if the nature of the work differs.
- Penalize vague, generic resumes that list soft skills (hardworking, positive attitude) or hobbies/personal profile sections without concrete work evidence.
- Do not reward experience in industries or functions that are clearly irrelevant to the JD, even if they share surface-level keywords.

Return ONLY a valid JSON object with this exact structure, no other text:
{{
  "job_role": "exact job title/role name from the JD (e.g. 'Senior Software Engineer', 'Sales Manager')",
  "candidate_name": "extracted full name or Unknown",
  "phone_number": "phone number in +91XXXXXXXXXX format (no space), or empty string if not found",
  "email": "email address or empty string if not found",
  "city": "candidate's current city or location from the resume, or empty string if not found",
  "overall_score": <integer 0-100>,
  "skills_score": <integer 0-40>,
  "experience_score": <integer 0-30>,
  "education_score": <integer 0-20>,
  "presentation_score": <integer 0-10>,
  "shortlisted": <true if overall_score >= 70, else false>,
  "job_hopper": <true if candidate has MORE THAN 3 years total experience AND more than 3 employers where tenure was under 1 year, else false>,
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap 1", "gap 2"],
  "summary": "2-3 sentence human readable summary of this candidate's fit"
}}

For phone_number: extract the candidate's phone number and format it as +91XXXXXXXXXX (10 digits after +91, no space). If the number is written without country code and appears to be Indian (10 digits starting with 6-9), prepend +91. If no phone number is found, return empty string.

For job_hopper: look at the candidate's work history. Count the number of employers where the listed tenure is less than 1 year (12 months). If the candidate has more than 3 years of total work experience AND more than 3 such short-tenure stints, set job_hopper to true. If total experience is 3 years or under, always set job_hopper to false regardless of tenure pattern.

Scoring rubric:
- skills_score (0-40): How well do the candidate's demonstrated skills match what the JD actually requires day-to-day? Required skills carry 3x more weight than preferred. Penalize if skills listed are generic soft skills with no supporting work evidence.
- experience_score (0-30): How closely does the candidate's past work match the nature and context of this role? Consider whether the type of work (not just the job title or industry) aligns. Irrelevant experience — even in the same industry — should score low. Reward specificity and depth over breadth. For candidates with a clear job-hopping pattern (more than 3 short stints under 1 year across a career of more than 3 years), apply an additional penalty of 5-8 points to reflect instability and lack of depth.
- education_score (0-20): Does education meet the role's requirements? Weight this appropriately — for roles that don't require specific degrees, don't over-penalize or over-reward based on field of study.
- presentation_score (0-10): Clarity, structure, and professionalism of the resume. Penalize resumes that are thin, vague, have no dates, or pad space with filler content.
- overall_score must equal skills_score + experience_score + education_score + presentation_score exactly.

JD: {jd_text}
RESUME: {resume_text}"""


def _build_prompt(jd_text: str, resume_text: str) -> str:
    return PROMPT_TEMPLATE.format(jd_text=jd_text, resume_text=resume_text)


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    return json.loads(cleaned)


def _error_result(filename: str, error: str) -> dict:
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    return {
        "job_role": "",
        "candidate_name": name,
        "phone_number": "",
        "email": "",
        "city": "",
        "filename": filename,
        "overall_score": 0,
        "skills_score": 0,
        "experience_score": 0,
        "education_score": 0,
        "presentation_score": 0,
        "shortlisted": False,
        "job_hopper": False,
        "strengths": [],
        "gaps": [],
        "summary": "",
        "error": error,
    }


async def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in environment.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, generation_config={"temperature": 0})
    response = await model.generate_content_async(prompt)
    return response.text


async def _call_claude(prompt: str) -> str:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in environment.")

    model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model=model_name,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


async def _call_with_backoff(call_fn, prompt: str, max_retries: int = 3) -> str:
    """Retry on rate-limit errors with exponential backoff + jitter."""
    for attempt in range(max_retries):
        try:
            return await call_fn(prompt)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(
                kw in err_str for kw in ["429", "rate", "quota", "exhausted", "temporarily"]
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)
            else:
                raise


async def score_resume(jd_text: str, resume_text: str, filename: str) -> dict:
    """
    Async: score a single resume against the job description.
    Uses a semaphore to cap concurrent LLM calls.
    Retries once on JSON parse failure. Returns an error dict on total failure.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    prompt = _build_prompt(jd_text, resume_text)

    async def call_llm() -> str:
        if provider == "gemini":
            return await _call_with_backoff(_call_gemini, prompt)
        elif provider == "claude":
            return await _call_with_backoff(_call_claude, prompt)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'gemini' or 'claude'.")

    async with _semaphore:
        last_error = None
        for attempt in range(2):
            try:
                raw = await call_llm()
                result = _extract_json(raw)
                result["filename"] = filename

                # Recompute overall_score from subscores so we don't trust LLM arithmetic.
                computed = (
                    result.get("skills_score", 0)
                    + result.get("experience_score", 0)
                    + result.get("education_score", 0)
                    + result.get("presentation_score", 0)
                )
                result["overall_score"] = computed
                # Job hoppers need a much stronger profile to be shortlisted.
                # The LLM already penalises them in experience_score; here we
                # additionally raise the shortlist bar so marginal candidates
                # with this pattern don't slip through.
                job_hopper = result.get("job_hopper", False)
                shortlist_threshold = 82 if job_hopper else 70
                result["shortlisted"] = computed >= shortlist_threshold

                # Fall back to filename stem if name is missing/generic.
                name = result.get("candidate_name", "").strip()
                if not name or name.lower() == "unknown":
                    result["candidate_name"] = filename.rsplit(".", 1)[0]

                return result
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error (attempt {attempt + 1}): {e}"
                continue
            except Exception as e:
                return _error_result(filename, f"LLM call failed: {e}")

        return _error_result(filename, last_error or "Failed to parse LLM response after 2 attempts.")
