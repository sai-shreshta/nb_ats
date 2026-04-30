import os
import sys

# Make ats/ importable so main.py can find parser and llm_client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ats"))

from main import app  # noqa: F401  – Vercel needs `app` in scope
