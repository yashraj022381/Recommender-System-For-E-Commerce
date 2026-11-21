"""The contract every recommender follows."""
import numpy as np

from abc import ABC, abstractmethod
from recsys.data.loader import Dataset

class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: Dataset) -> "Recommender":
        ...


    @abstractmethod
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Return a (len(user_ids), n_items) array. Higher = better."""


    def recommend(self, user_id: int, k: int = 10, exclude=None) -> list[int]:
        """Top-k item ids for one user, skipping `exclude` (items already seen)."""
        s = np.array(self.score(np.array([user_id]))[0], dtype=np.float32, copy=True)
        if exclude is not None and len(exclude):
            s[list(exclude)] = -np.inf
        top = np.argpartition(-s, k - 1)[:k]
        return top[np.argsort(-s[top])].tolist()
