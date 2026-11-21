"""Optional LLM access (Groq). Everything in the project works without it."""
import os

MODEL = os.environ.get("RECSYS_LLM_MODEL", "openai/gpt-oss-120b")


def make_client():
    """Groq client, or None when there is no API key or no package (callers then use fallbacks)."""
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq
    except ImportError:
        return None
    return Groq(timeout=30.0, max_retries=2)
