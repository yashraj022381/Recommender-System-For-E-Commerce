"""Vector search. FAISS if installed, exact NumPy search otherwise."""
import numpy as np


class VectorIndex:
    """Inner-product search over item codes (codes are L2-normalized, so IP = cosine).

    IndexFlatIP is exact. For millions of items swap in faiss.IndexHNSWFlat or an IVF index.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.backend = "faiss"
        except ImportError:
            self.index, self.backend = None, "numpy"

    def search(self, queries: np.ndarray, k: int):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, len(self.vectors))
        if self.index is not None:
            return self.index.search(q, k)
        S = q @ self.vectors.T
        I = np.argpartition(-S, k - 1, axis=1)[:, :k]
        D = np.take_along_axis(S, I, 1)
        o = np.argsort(-D, axis=1)
        return np.take_along_axis(D, o, 1), np.take_along_axis(I, o, 1)
