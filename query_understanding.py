import re
from typing import Dict, List

def extract_entities(text: str) -> List[str]:
    # super-light entity extraction: hashtags, @mentions, capitalized word sequences
    ents = set()
    for m in re.findall(r"(#[\w-]+|@[\w-]+)", text):
        ents.add(m)
    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
        ents.add(m.strip())
    return sorted(list(ents))

def build_queries(user_text: str, read_types: List[str], scope: str) -> Dict:
    entities = extract_entities(user_text)

    # IMPORTANT: keep semantic_query clean for embeddings
    semantic_query = user_text.strip()

    # Keyword query for SQLite LIKE retrieval
    keyword_query = " ".join(
        [w for w in re.findall(r"[A-Za-z0-9#@_-]+", user_text.lower()) if len(w) > 2]
    )

    return {
        "semantic_query": semantic_query,
        "keyword_query": keyword_query,
        "entities": entities,
        "filters": {
            "types": read_types,
            "scope_prefix": scope
        }
    }
