import json
import re
from typing import Dict, List
from llm_client import OllamaLLM

ALLOWED_TYPES = {"preference", "commitment", "project_state", "episodic"}

SYSTEM_PROMPT = """
You are a routing module for a long-term memory assistant.

You must classify the USER_TEXT and output STRICT JSON only.

Allowed memory types:
- preference
- commitment
- project_state
- episodic

Rules:
- needs_memory = true if the user refers to earlier context, previous decisions, short follow-ups (<=6 words), or unclear references.
- read_types = minimal memory types required to answer correctly.
- write_types = memory types that should be stored long-term.
- If unsure, lean conservative but safe.

Return ONLY valid JSON:
{
  "needs_memory": true/false,
  "read_types": ["..."],
  "write_types": ["..."],
  "confidence": 0.0-1.0
}
"""

_llm = OllamaLLM(model="llama3.2:3b")


# -----------------------------
# Fallback heuristic (important!)
# -----------------------------
def _fallback_router(user_text: str) -> Dict:
    t = user_text.lower()
    needs_memory = len(t.split()) <= 6 or any(
        k in t for k in ["remember", "earlier", "last time", "you said", "continue", "resume"]
    )

    read_types: List[str] = []
    if any(k in t for k in ["prefer", "like", "hate", "format"]):
        read_types.append("preference")
    if any(k in t for k in ["deadline", "commit", "i will", "we will"]):
        read_types.append("commitment")
    if any(k in t for k in ["project", "architecture", "repo", "code", "hackathon"]):
        read_types.append("project_state")
    if not read_types:
        read_types = ["project_state", "episodic"]

    write_types: List[str] = []
    if any(k in t for k in ["i prefer", "i like", "i hate", "from now on"]):
        write_types.append("preference")
    if any(k in t for k in ["i will", "we will", "by tomorrow", "deadline"]):
        write_types.append("commitment")

    return {
        "needs_memory": needs_memory,
        "read_types": read_types,
        "write_types": write_types,
        "confidence": 0.55,
    }


def _safe_json_parse(s: str) -> Dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?", "", s)
    s = re.sub(r"```$", "", s)
    return json.loads(s)


def _sanitize(output: Dict) -> Dict:
    needs_memory = bool(output.get("needs_memory", False))

    read_types = output.get("read_types", [])
    if not isinstance(read_types, list):
        read_types = []
    read_types = [t for t in read_types if t in ALLOWED_TYPES]
    if not read_types:
        read_types = ["project_state", "episodic"]

    write_types = output.get("write_types", [])
    if not isinstance(write_types, list):
        write_types = []
    write_types = [t for t in write_types if t in ALLOWED_TYPES]

    confidence = output.get("confidence", 0.6)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    return {
        "needs_memory": needs_memory,
        "read_types": read_types,
        "write_types": write_types,
        "confidence": confidence,
    }


# -----------------------------
# Main router
# -----------------------------
def route_turn(user_text: str) -> Dict:
    if not user_text.strip():
        return _fallback_router(user_text)

    try:
        prompt = f"USER_TEXT:\n{user_text}"
        raw = _llm.chat(system=SYSTEM_PROMPT, user=prompt, timeout_s=20)

        parsed = _safe_json_parse(raw)
        cleaned = _sanitize(parsed)

        # If model unsure, fallback
        if cleaned["confidence"] < 0.4:
            return _fallback_router(user_text)

        # Short follow-ups should usually use memory
        if len(user_text.split()) <= 6:
            cleaned["needs_memory"] = True

        return cleaned

    except Exception:
        return _fallback_router(user_text)
