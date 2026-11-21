import json

import pandas as pd
import numpy as np

from dataclasses import dataclass
from functools import cached_property
from scipy import sparse as sp

from recsys.config import PATHS



@dataclass
class Dataset:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame
    n_users: int
    n_items: int


    def csr(self, df: pd.DataFrame, positives_only: bool = False) -> sp.csr_matrix:
        """Binary user x item matrix from an interactions table."""
        if positives_only:
            df = df[df["label"] == 1]
        return sp.csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df["user_id"], df["item_id"])),
            shape=(self.n_users, self.n_items),
        )


    @cached_property
    def train_csr(self) -> sp.csr_matrix:
        return self.csr(self.train)

    @cached_property
    def train_pos_csr(self) -> sp.csr_matrix:
        return self.csr(self.train, positives_only=True)



def load_dataset() -> Dataset:
    p = PATHS.processed
    meta = json.loads((p / "meta.json").read_text())
    return Dataset(
        train=pd.read_parquet(p / "train.parquet"),
        val=pd.read_parquet(p / "val.parquet"),
        test=pd.read_parquet(p / "test.parquet"),
        items=pd.read_parquet(p / "items.parquet"),
        n_users=meta["n_users"],
        n_items=meta["n_items"],
    )
    
