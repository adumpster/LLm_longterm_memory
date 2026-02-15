from typing import Any, Dict, List, Tuple, Optional

def choose_profile_keys(user_text: str) -> List[str]:
    t = user_text.lower()
    keys = []
    if any(k in t for k in ["format", "bullet", "concise"]):
        keys.append("preferred_format")
    if any(k in t for k in ["language", "python", "cpp", "c++"]):
        keys.append("preferred_language")
    if any(k in t for k in ["project", "hackathon", "memory", "architecture"]):
        keys.append("current_project")
    return keys

def build_memory_bundle(
    profile_items: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    latest_summary: Optional[Dict[str, Any]],
    max_items: int = 8,
) -> Tuple[str, List[str]]:
    """
    Returns: (bundle_text, memory_ids_used)
    """
    lines: List[str] = []
    used_ids: List[str] = []

    lines.append("ACTIVE MEMORY (use as factual; do not invent missing memories):")

    # profile first (0-2)
    for p in profile_items[:2]:
        lines.append(f"- [PROFILE|key={p['key']}|conf={p['confidence']:.2f}] {p['value']}")
        used_ids.append(f"profile:{p['key']}")

    # summary optionally (0-1)
    if latest_summary and len(lines) < (max_items + 1):
        lines.append(f"- [SUMMARY|scope={latest_summary['scope']}] {latest_summary['summary_text']}")
        used_ids.append(latest_summary["id"])

    # retrieved memories (fill remaining)
    remaining = max_items - (len(lines) - 1)
    for m in retrieved[: max(0, remaining)]:
        lines.append(
            f"- [ID={m['id']}|type={m['type']}|score={m['_score']:.3f}|rel={m['_rel']:.3f}|rec={m['_rec']:.3f}] {m['text']}"
        )
        used_ids.append(m["id"])

    bundle = "\n".join(lines).strip()
    return bundle, used_ids
