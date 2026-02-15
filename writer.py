import uuid
from typing import Any, Dict, List

from db import MemoryDB, now_ts
from embedder import Embedder
from vector_index import VectorIndex
from query_understanding import extract_entities


def new_id() -> str:
    return str(uuid.uuid4())


def extract_memories_rule_based(
    user_text: str,
    assistant_text: str,
    scope: str,
    write_types: List[str],
) -> List[Dict[str, Any]]:
    t = user_text.strip()
    low = t.lower()
    ents = extract_entities(t)

    recs: List[Dict[str, Any]] = []

    # --- preference ---
    if (
        "preference" in write_types
        or any(k in low for k in ["i prefer", "i like", "i hate", "from now on", "going forward"])
    ):
        recs.append(
            {
                "id": new_id(),
                "type": "preference",
                "scope": "global",
                "text": f"User preference stated: {t}",
                "entities": ents,
                "importance": 0.7,
                "confidence": 0.8,
                "created_at": now_ts(),
                "last_seen_at": now_ts(),
                "source_turns": [],
            }
        )

    # --- commitment ---
    if (
        "commitment" in write_types
        or any(k in low for k in ["i will", "we will", "i'll", "we'll", "by tomorrow", "by monday"])
    ):
        recs.append(
            {
                "id": new_id(),
                "type": "commitment",
                "scope": scope,
                "text": f"Commitment mentioned: {t}",
                "entities": ents,
                "importance": 0.8,
                "confidence": 0.75,
                "created_at": now_ts(),
                "last_seen_at": now_ts(),
                "source_turns": [],
            }
        )

    # --- project_state ---
    # store stable project intent/decisions with long TTL
    if any(k in low for k in ["we are building", "we're building", "our architecture", "the project", "hackathon"]):
        recs.append(
            {
                "id": new_id(),
                "type": "project_state",
                "scope": scope,
                "text": f"Project state / decision: {t}",
                "entities": ents,
                "importance": 0.75,
                "confidence": 0.75,
                "created_at": now_ts(),
                "last_seen_at": now_ts(),
                "source_turns": [],
                "ttl_days": 365,
            }
        )

    # --- episodic ---
    # keep it short-lived and low importance so it doesn't dominate retrieval
    if len(t.split()) >= 10 and not any(k in low for k in ["remember this", "save this"]):
        recs.append(
            {
                "id": new_id(),
                "type": "episodic",
                "scope": scope,
                "text": f"Conversation detail: user said '{t[:240]}'",
                "entities": ents,
                "importance": 0.35,
                "confidence": 0.7,
                "created_at": now_ts(),
                "last_seen_at": now_ts(),
                "source_turns": [],
                "ttl_days": 14,
            }
        )

    return recs


def upsert_profile_from_text(db: MemoryDB, user_text: str) -> None:
    low = user_text.lower()

    # basic preference capture (you can add more keys here)
    if "bullet" in low or "concise" in low:
        db.upsert_profile("preferred_format", {"style": "concise_bullets"}, confidence=0.85)

    if "python" in low:
        db.upsert_profile("preferred_language", {"language": "python"}, confidence=0.8)

    if "c++" in low or "cpp" in low:
        db.upsert_profile("preferred_language", {"language": "cpp"}, confidence=0.8)


def write_memories(
    db: MemoryDB,
    embedder: Embedder,
    vindex: VectorIndex,
    memories: List[Dict[str, Any]],
) -> None:
    if not memories:
        return

    # which types we embed into FAISS (others may be stored but not embedded)
    embed_types = {"episodic", "project_state", "preference", "commitment"}

    texts_to_embed: List[str] = []
    idx_map: List[str] = []

    for m in memories:
        if m.get("type") in embed_types:
            texts_to_embed.append(f"{m['type']} | {m['scope']} | {m['text']}")
            idx_map.append(m["id"])

    vecs = embedder.embed(texts_to_embed) if texts_to_embed else None

    # Insert rows first
    for m in memories:
        db.insert_memory(m)

    # Attach embeddings
    if vecs is not None:
        for i, mem_id in enumerate(idx_map):
            emb_id = vindex.add(vecs[i])
            db.update_memory_embedding_id(mem_id, emb_id)

    vindex.save()
