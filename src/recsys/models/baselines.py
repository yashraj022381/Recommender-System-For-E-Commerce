"""Two sanity-check robots."""
import numpy as np

from recsys.data.loader import Dataset
from recsys.models.base import Recommender



class RandomRec(Recommender):
    name = "Random"


    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)


    def fit(self, data: Dataset):
        self.n_items = data.n_items
        return self

    def score(self, user_ids):
        return self.rng.random((len(user_ids), self.n_items), dtype=np.float32)


class PopularityRec(Recommender):
    name = "Popularity"

    def fit(self, data: Dataset):
        # popularity = number of positive (liked) interactions in TRAIN only
        self.pop = np.asarray(data.train_pos_csr.sum(0)).ravel().astype(np.float32)
        return self


    def score(self, user_ids):
        return np.broadcast_to(self.pop, (len(user_ids), len(self.pop)))
