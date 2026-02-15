import os
import numpy as np
from typing import List, Tuple


class VectorIndex:
    """
    FAISS IndexFlatIP (inner product) for cosine similarity.
    We store normalized embeddings so inner product == cosine.
    """

    def __init__(self, dim: int, index_path: str = "faiss.index"):
        self.dim = int(dim)
        self.index_path = index_path
        self._faiss = None
        self.index = None
        self._load_faiss()

    # ------------------------------
    # Load FAISS safely
    # ------------------------------
    def _load_faiss(self):
        import faiss  # type: ignore

        self._faiss = faiss

        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)

                # CRITICAL FIX:
                # If embedding dimension changed (model switch),
                # rebuild index automatically.
                if int(self.index.d) != self.dim:
                    print(
                        f"[VectorIndex] Dimension mismatch "
                        f"(index={self.index.d}, expected={self.dim}). "
                        f"Rebuilding index."
                    )
                    self.index = faiss.IndexFlatIP(self.dim)

            except Exception:
                # If index corrupted, rebuild safely
                print("[VectorIndex] Failed to load index. Rebuilding new index.")
                self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    # ------------------------------
    # Save index
    # ------------------------------
    def save(self) -> None:
        if self.index is not None:
            self._faiss.write_index(self.index, self.index_path)

    # ------------------------------
    # Normalize vectors
    # ------------------------------
    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    # ------------------------------
    # Add single vector
    # ------------------------------
    def add(self, vec: np.ndarray) -> int:
        """
        Adds a single vector.
        Returns embedding_id (row id in FAISS).
        """
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        if vec.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {vec.shape[1]}"
            )

        vec = self._normalize(vec)

        before = self.index.ntotal
        self.index.add(vec)
        return int(before)

    # ------------------------------
    # Search vectors
    # ------------------------------
    def search(self, query_vec: np.ndarray, topk: int = 20) -> Tuple[List[int], List[float]]:
        """
        Returns (embedding_ids, similarities) sorted high->low.
        """

        if self.index is None or self.index.ntotal == 0:
            return [], []

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        query_vec = self._normalize(query_vec)

        sims, ids = self.index.search(query_vec, topk)

        emb_ids = [int(i) for i in ids[0] if i != -1]
        sim_scores = [float(s) for s in sims[0][: len(emb_ids)]]

        return emb_ids, sim_scores
