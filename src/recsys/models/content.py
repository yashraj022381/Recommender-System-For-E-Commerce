"""Content-based recommenders: user profile = average of liked items' text vectors."""
import numpy as np

from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.data.loader import Dataset
from recsys.models.base import Recommender
from sentence_transformers import SentenceTransformer     # lazy import



class _Content(Recommender):
    """Subclasses set self.X (items x features, L2-normalized, sparse or dense)."""

    def _prepare(self, data: Dataset):
        self.R = data.train_pos_csr.astype(np.float32)
        self.items = data.items
        pop = np.asarray(self.R.sum(0)).ravel().astype(np.float32)
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1.0)

    def _profiles(self, user_ids):
        """Average the vectors of each user's liked items -> (n_users, features)."""
        R = self.R[user_ids]
        counts = np.asarray(R.sum(1)).ravel()
        counts[counts == 0] = 1.0                       # users with no likes -> zero profile
        P = sp.diags(1.0 / counts) @ R @ self.X
        return P

    def score(self, user_ids):
        P = self._profiles(user_ids)
        S = P @ self.X.T                                # (users, items) similarity
        S = S.toarray() if sp.issparse(S) else np.asarray(S)
        return S + self.tiebreak

    def explain(self, user_id: int, item_id: int, n: int = 3):
        """Which liked items look most like this item (by text)? -> [(item_id, cosine)]"""
        liked = self.R[user_id].indices
        if len(liked) == 0:
            return []
        t = self.X[item_id]
        sims = self.X[liked] @ t.T
        sims = sims.toarray().ravel() if sp.issparse(sims) else np.asarray(sims).ravel()
        order = np.argsort(-sims)[:n]
        return [(int(liked[j]), float(sims[j])) for j in order]


class TfidfContentRec(_Content):
    def __init__(self, max_features: int = 20000, ngram_range=(1, 2), min_df: int = 1):
        self.max_features, self.ngram_range, self.min_df = max_features, ngram_range, min_df
        self.name = "Content-TFIDF"

    def fit(self, data: Dataset):
        self._prepare(data)
        text = (self.items["title"].fillna("") + " " + self.items["text"].fillna("")).tolist()
        self.vec = TfidfVectorizer(
            stop_words="english", max_features=self.max_features,
            ngram_range=self.ngram_range, min_df=self.min_df, sublinear_tf=True,
        )
        self.X = normalize(self.vec.fit_transform(text)).astype(np.float32).tocsr()
        return self

    def top_terms(self, user_id: int, item_id: int, n: int = 5):
        """Words that make this item match the user's profile (for 'Why this?')."""
        P = self._profiles(np.array([user_id]))
        contrib = P.multiply(self.X[item_id]).toarray().ravel()
        vocab = np.array(self.vec.get_feature_names_out())
        top = np.argsort(-contrib)[:n]
        return [(str(vocab[j]), float(contrib[j])) for j in top if contrib[j] > 0]


class EmbeddingContentRec(_Content):
    """Sentence embeddings (optional; downloads a small model on first run)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64):
        self.model_name, self.batch_size = model_name, batch_size
        self.name = "Content-Embed"

    def fit(self, data: Dataset):
        self._prepare(data)
        text = (self.items["title"].fillna("") + ". " + self.items["text"].fillna("")).tolist()
        enc = SentenceTransformer(self.model_name)
        self.X = enc.encode(text, batch_size=self.batch_size, normalize_embeddings=True,
                            show_progress_bar=False).astype(np.float32)
        return self
