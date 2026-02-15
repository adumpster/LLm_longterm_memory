import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

# -----------------------------
# SCHEMA
# -----------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  scope TEXT NOT NULL,
  text TEXT NOT NULL,
  entities TEXT NOT NULL,
  importance REAL NOT NULL,
  confidence REAL NOT NULL,
  created_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  source_turns TEXT NOT NULL,
  embedding_id INTEGER,
  is_active INTEGER NOT NULL DEFAULT 1,
  ttl_days INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memories_type_scope ON memories(type, scope);
CREATE INDEX IF NOT EXISTS idx_memories_scope_lastseen ON memories(scope, last_seen_at);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding_id);

CREATE TABLE IF NOT EXISTS profile_kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  confidence REAL NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  summary_text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  embedding_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_summaries_scope_created ON summaries(scope, created_at);

CREATE TABLE IF NOT EXISTS turn_logs (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  user_text TEXT NOT NULL,
  assistant_text TEXT NOT NULL,
  memory_ids_used TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turn_logs_session ON turn_logs(session_id, turn_id);
"""

def now_ts() -> int:
    return int(time.time())


# -----------------------------
# MEMORY DB
# -----------------------------

class MemoryDB:
    def __init__(self, path: str = "memory.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # =====================================================
    # PROFILE KV
    # =====================================================

    def upsert_profile(self, key: str, value: Any, confidence: float) -> None:
        v = json.dumps(value, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO profile_kv(key, value, confidence, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value=excluded.value,
              confidence=excluded.confidence,
              updated_at=excluded.updated_at
            """,
            (key, v, float(confidence), now_ts()),
        )
        self.conn.commit()

    def get_profile_keys(self, keys: List[str]) -> List[Dict[str, Any]]:
        if not keys:
            return []

        q = f"""
        SELECT key, value, confidence, updated_at
        FROM profile_kv
        WHERE key IN ({",".join(["?"] * len(keys))})
        """
        rows = self.conn.execute(q, keys).fetchall()

        return [
            {
                "key": r["key"],
                "value": json.loads(r["value"]),
                "confidence": float(r["confidence"]),
                "updated_at": int(r["updated_at"]),
            }
            for r in rows
        ]

    def list_profile(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM profile_kv").fetchall()
        return [
            {
                "key": r["key"],
                "value": json.loads(r["value"]),
                "confidence": float(r["confidence"]),
                "updated_at": int(r["updated_at"]),
            }
            for r in rows
        ]

    # =====================================================
    # MEMORY INSERT / UPDATE
    # =====================================================

    def insert_memory(self, rec: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO memories(
              id, type, scope, text, entities,
              importance, confidence,
              created_at, last_seen_at,
              source_turns, embedding_id,
              is_active, ttl_days
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec["id"],
                rec["type"],
                rec["scope"],
                rec["text"],
                json.dumps(rec.get("entities", []), ensure_ascii=False),
                float(rec.get("importance", 0.5)),
                float(rec.get("confidence", 0.8)),
                int(rec.get("created_at", now_ts())),
                int(rec.get("last_seen_at", now_ts())),
                json.dumps(rec.get("source_turns", []), ensure_ascii=False),
                rec.get("embedding_id", None),
                int(rec.get("is_active", 1)),
                rec.get("ttl_days", None),
            ),
        )
        self.conn.commit()

    def update_memory_embedding_id(self, mem_id: str, embedding_id: int) -> None:
        self.conn.execute(
            "UPDATE memories SET embedding_id=? WHERE id=?",
            (int(embedding_id), mem_id),
        )
        self.conn.commit()

    def touch_memories(self, mem_ids: List[str]) -> None:
        if not mem_ids:
            return

        q = f"""
        UPDATE memories
        SET last_seen_at=?
        WHERE id IN ({",".join(["?"] * len(mem_ids))})
        """
        self.conn.execute(q, [now_ts(), *mem_ids])
        self.conn.commit()

    # =====================================================
    # TTL-ENFORCED FETCH
    # =====================================================

    def _ttl_clause(self) -> str:
        return "(ttl_days IS NULL OR created_at >= (strftime('%s','now') - ttl_days*86400))"

    def fetch_memories_by_embedding_ids(
        self,
        embedding_ids: List[int],
        types: Optional[List[str]] = None,
        scope_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not embedding_ids:
            return []

        clauses = [
            f"embedding_id IN ({','.join(['?'] * len(embedding_ids))})",
            "is_active=1",
            self._ttl_clause(),
        ]
        params: List[Any] = [*embedding_ids]

        if types:
            clauses.append(f"type IN ({','.join(['?'] * len(types))})")
            params.extend(types)

        if scope_prefix:
            clauses.append("scope LIKE ?")
            params.append(f"{scope_prefix}%")

        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)}"
        rows = self.conn.execute(sql, params).fetchall()

        return [_row_to_memory(r) for r in rows]

    def fetch_recent_memories(
        self,
        scope_prefix: str,
        limit: int = 50,
        types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        clauses = [
            "scope LIKE ?",
            "is_active=1",
            self._ttl_clause(),
        ]
        params: List[Any] = [f"{scope_prefix}%"]

        if types:
            clauses.append(f"type IN ({','.join(['?'] * len(types))})")
            params.extend(types)

        sql = f"""
        SELECT * FROM memories
        WHERE {' AND '.join(clauses)}
        ORDER BY last_seen_at DESC
        LIMIT ?
        """
        params.append(int(limit))

        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_memory(r) for r in rows]

    def search_memories_keyword(
        self,
        keywords: List[str],
        scope_prefix: str,
        limit: int = 50,
        types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        keywords = [k.lower().strip() for k in keywords if k.strip()]
        if not keywords:
            return []

        clauses = [
            "scope LIKE ?",
            "is_active=1",
            self._ttl_clause(),
        ]
        params: List[Any] = [f"{scope_prefix}%"]

        if types:
            clauses.append(f"type IN ({','.join(['?'] * len(types))})")
            params.extend(types)

        like_parts = []
        for k in keywords[:8]:
            like_parts.append("LOWER(text) LIKE ?")
            params.append(f"%{k}%")

        clauses.append("(" + " OR ".join(like_parts) + ")")

        sql = f"""
        SELECT * FROM memories
        WHERE {' AND '.join(clauses)}
        ORDER BY last_seen_at DESC
        LIMIT ?
        """
        params.append(int(limit))

        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_memory(r) for r in rows]

    # =====================================================
    # SUMMARIES
    # =====================================================

    def insert_summary(self, rec: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO summaries(id, scope, summary_text, created_at, embedding_id)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                rec["id"],
                rec["scope"],
                rec["summary_text"],
                int(rec.get("created_at", now_ts())),
                rec.get("embedding_id", None),
            ),
        )
        self.conn.commit()

    def latest_summary(self, scope: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE scope=? ORDER BY created_at DESC LIMIT 1",
            (scope,),
        ).fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "scope": row["scope"],
            "summary_text": row["summary_text"],
            "created_at": int(row["created_at"]),
            "embedding_id": row["embedding_id"],
        }

    # =====================================================
    # TURN LOGS
    # =====================================================

    def log_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        memory_ids_used: List[str],
    ) -> None:

        self.conn.execute(
            """
            INSERT INTO turn_logs(session_id, user_text, assistant_text, memory_ids_used, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (session_id, user_text, assistant_text, json.dumps(memory_ids_used), now_ts()),
        )
        self.conn.commit()

    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM turn_logs WHERE session_id=? ORDER BY turn_id ASC",
            (session_id,),
        ).fetchall()

        return [
            {
                "turn_id": int(r["turn_id"]),
                "user_text": r["user_text"],
                "assistant_text": r["assistant_text"],
                "memory_ids_used": json.loads(r["memory_ids_used"]),
                "created_at": int(r["created_at"]),
            }
            for r in rows
        ]


def _row_to_memory(r: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "type": r["type"],
        "scope": r["scope"],
        "text": r["text"],
        "entities": json.loads(r["entities"]) if r["entities"] else [],
        "importance": float(r["importance"]),
        "confidence": float(r["confidence"]),
        "created_at": int(r["created_at"]),
        "last_seen_at": int(r["last_seen_at"]),
        "source_turns": json.loads(r["source_turns"]) if r["source_turns"] else [],
        "embedding_id": r["embedding_id"],
        "is_active": int(r["is_active"]),
        "ttl_days": r["ttl_days"],
    }
