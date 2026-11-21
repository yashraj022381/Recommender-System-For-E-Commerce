"""One place for the cold-user rule: no history -> popularity."""
import numpy as np

from recsys.data.loader import Dataset
from recsys.models.base import Recommender



class ColdUserFallback(Recommender):
    def __init__(self, model: Recommender, data: Dataset):
        self.model = model
        self.name = model.name
        R = data.train_pos_csr
        pop = np.asarray(R.sum(0)).ravel().astype(np.float32)
        self.pop = pop / max(pop.max(), 1.0)
        self.has_history = np.asarray(R.sum(1)).ravel() > 0


    def fit(self, data):
        self.model.fit(data)
        return self

    def score(self, user_ids):
        user_ids = np.asarray(user_ids)
        s = np.array(self.model.score(user_ids), dtype=np.float32, copy=True)
        cold = ~self.has_history[user_ids]
        if cold.any():
            s[cold] = self.pop
        return s

    def __getattr__(self, key):                  # forward explain(), etc. to the wrapped model
        if key == "model":
            raise AttributeError(key)
        return getattr(self.model, key)

    
