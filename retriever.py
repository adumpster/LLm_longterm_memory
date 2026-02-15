import math
import time
from typing import Any, Dict, List, Optional

from db import MemoryDB
from embedder import Embedder
from vector_index import VectorIndex

def recency_score(last_seen_at: int, tau_days: float = 7.0) -> float:
    dt = max(0, int(time.time()) - int(last_seen_at))
    days = dt / 86400.0
    return float(math.exp(-days / tau_days))

def rank_candidates(
    candidates: List[Dict[str, Any]],
    similarities: Dict[int, float],
    query_entities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    ranked = []
    for m in candidates:
        rel = 0.0
        if m.get("embedding_id") is not None:
            rel = float(similarities.get(int(m["embedding_id"]), 0.0))
        rec = recency_score(m["last_seen_at"])
        imp = float(m.get("importance", 0.5))
        conf = float(m.get("confidence", 0.8))

        ent_bonus = 0.0
        if query_entities:
            mem_ents = set([str(e).lower() for e in (m.get("entities") or [])])
            q_ents = set([str(e).lower() for e in query_entities])
            if mem_ents and q_ents and (mem_ents & q_ents):
                ent_bonus = 0.06

        score = 0.52 * rel + 0.25 * rec + 0.15 * imp + 0.05 * conf + ent_bonus

        mm = dict(m)
        mm["_score"] = score
        mm["_rel"] = rel
        mm["_rec"] = rec
        ranked.append(mm)

    ranked.sort(key=lambda x: x["_score"], reverse=True)
    return ranked

def dedupe_by_text(items: List[Dict[str, Any]], max_items: int = 50) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = it["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= max_items:
            break
    return out

class Retriever:
    def __init__(self, db: MemoryDB, embedder: Embedder, vindex: VectorIndex):
        self.db = db
        self.embedder = embedder
        self.vindex = vindex

    def retrieve(
        self,
        semantic_query: str,
        keyword_query: str,
        entities: List[str],
        types: List[str],
        scope_prefix: str,
        topk: int = 50,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        # 1) Semantic retrieval (FAISS)
        sim_map: Dict[int, float] = {}
        try:
            qvec = self.embedder.embed([semantic_query])[0]
            emb_ids, sims = self.vindex.search(qvec, topk=max(30, topk))
            sim_map = {emb_id: float(s) for emb_id, s in zip(emb_ids, sims)}
            candidates.extend(
                self.db.fetch_memories_by_embedding_ids(emb_ids, types=types, scope_prefix=scope_prefix)
            )
        except Exception:
            sim_map = {}

        # 2) Keyword backstop
        kw_tokens = [t for t in keyword_query.split() if t]
        if kw_tokens:
            candidates.extend(
                self.db.search_memories_keyword(kw_tokens, scope_prefix=scope_prefix, limit=topk * 3, types=types)
            )

        # 3) Recency backstop
        candidates.extend(
            self.db.fetch_recent_memories(scope_prefix=scope_prefix, limit=topk * 3, types=types)
        )

        # Dedup by ID first
        by_id: Dict[str, Dict[str, Any]] = {}
        for m in candidates:
            by_id[m["id"]] = m
        unique = list(by_id.values())

        ranked = rank_candidates(unique, similarities=sim_map, query_entities=entities)
        return dedupe_by_text(ranked, max_items=topk)
