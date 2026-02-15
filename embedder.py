import os
import json
import numpy as np
import urllib.request
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class Embedder:
    """
    Primary: sentence-transformers
    Fallback: Ollama /api/embeddings (local)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ollama_base_url: str = "http://localhost:11434",
        ollama_embed_model: str = "nomic-embed-text",
    ):
        os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_embed_model = ollama_embed_model

        self._st_model = None
        if SentenceTransformer is not None:
            try:
                self._st_model = SentenceTransformer(model_name)
            except Exception:
                self._st_model = None

    def _embed_ollama(self, texts: List[str], timeout_s: int = 60) -> np.ndarray:
        url = f"{self.ollama_base_url}/api/embeddings"
        out_vecs = []
        for t in texts:
            payload = {"model": self.ollama_embed_model, "prompt": t}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            out_vecs.append(obj["embedding"])

        vecs = np.asarray(out_vecs, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype="float32")

        if self._st_model is not None:
            vecs = self._st_model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype="float32")

        return self._embed_ollama(texts)
