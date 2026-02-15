from typing import Dict, List
from llm_client import OllamaLLM

from db import MemoryDB
from embedder import Embedder
from vector_index import VectorIndex
from router import route_turn
from query_understanding import build_queries
from retriever import Retriever
from bundler import choose_profile_keys, build_memory_bundle
from writer import extract_memories_rule_based, upsert_profile_from_text, write_memories


class MemorySystem:
    def __init__(self, db_path: str = "memory.db", index_path: str = "faiss.index"):
        self.db = MemoryDB(db_path)
        self.embedder = Embedder()
        self.llm = OllamaLLM(model="llama3.2:3b")

        # get embedding dim from model by embedding a dummy string
        dim = int(self.embedder.embed(["dim probe"]).shape[1])
        self.vindex = VectorIndex(dim=dim, index_path=index_path)
        self.retriever = Retriever(self.db, self.embedder, self.vindex)

    def _generate_response(self, user_text: str, memory_bundle: str, recent_ctx: str) -> str:
        system = (
            "You are a helpful assistant.\n"
            "You are given:\n"
            "1) RECENT CHAT CONTEXT (short-term, last few turns).\n"
            "2) ACTIVE MEMORY (long-term retrieved memories).\n\n"
            "Rules:\n"
            "- Answer normally if the question is self-sufficient.\n"
            "- Use RECENT CHAT CONTEXT for follow-ups (e.g., 'I like chocolate' -> suggest chocolate cake).\n"
            "- Use ACTIVE MEMORY for older facts and persistent preferences.\n"
            "- If needed info is missing from both, ask ONE targeted question.\n"
            "- Never mention internal terms like ACTIVE MEMORY or RECENT CHAT CONTEXT to the user.\n"
        )

        user = (
            f"RECENT CHAT CONTEXT:\n{recent_ctx if recent_ctx else '(none)'}\n\n"
            f"ACTIVE MEMORY:\n{memory_bundle if memory_bundle else '(none)'}\n\n"
            f"USER QUESTION:\n{user_text}\n"
        )

        try:
            return self.llm.chat(system=system, user=user)
        except Exception as e:
            return self._fallback_answer(user_text, memory_bundle, error=str(e))

    @staticmethod
    def _fallback_answer(user_text: str, bundle: str, error: str = "") -> str:
        lines = []
        lines.append("⚠️ I couldn't reach the local Ollama model, so I'm in fallback mode.")
        if error:
            lines.append(f"Reason: {error}")
        lines.append("")
        lines.append("Here is the memory I would have used:")
        lines.append(bundle if bundle else "(No memory retrieved)")
        lines.append("")
        lines.append("Start Ollama with: `ollama serve` and retry.")
        lines.append("")
        lines.append("Your question was:")
        lines.append(user_text)
        return "\n".join(lines)

    def handle_turn(
        self,
        session_id: str,
        user_text: str,
        scope: str = "project:hackathon",
        topk: int = 50,
    ) -> Dict:
        routing = route_turn(user_text)

        # ✅ Always include short-term context, regardless of memory routing
        recent_turns = self.db.load_recent_turns(session_id, limit=8)
        recent_ctx_lines: List[str] = []
        for t in recent_turns:
            recent_ctx_lines.append(f"USER: {t['user_text']}")
            recent_ctx_lines.append(f"ASSISTANT: {t['assistant_text']}")
        recent_ctx = "\n".join(recent_ctx_lines)

        bundle = ""
        used_ids: List[str] = []

        # Retrieve long-term memory only when router says so
        if routing.get("needs_memory", False):
            q = build_queries(user_text, routing["read_types"], scope=scope)

            # profile
            pkeys = choose_profile_keys(user_text)
            profile_items = self.db.get_profile_keys(pkeys)

            # summary
            latest_summary = self.db.latest_summary(scope)

            # retrieved memories (hybrid signature)
            retrieved = self.retriever.retrieve(
                semantic_query=q["semantic_query"],
                keyword_query=q["keyword_query"],
                entities=q["entities"],
                types=q["filters"]["types"],
                scope_prefix=q["filters"]["scope_prefix"],
                topk=topk,
            )

            bundle, used_ids = build_memory_bundle(profile_items, retrieved, latest_summary, max_items=8)

            # touch used memories (only actual memory ids)
            touch_ids = [mid for mid in used_ids if not mid.startswith("profile:")]
            self.db.touch_memories(touch_ids)

        # ✅ Generate response using BOTH recent context and optional long-term memory
        assistant_text = self._generate_response(user_text, bundle, recent_ctx)

        # write pipeline
        upsert_profile_from_text(self.db, user_text)
        new_mems = extract_memories_rule_based(user_text, assistant_text, scope, routing.get("write_types", []))
        write_memories(self.db, self.embedder, self.vindex, new_mems)

        # log turn for replay
        self.db.log_turn(session_id, user_text, assistant_text, used_ids)

        return {
            "routing": routing,
            "memory_bundle": bundle,
            "memory_ids_used": used_ids,
            "assistant_text": assistant_text,
        }
