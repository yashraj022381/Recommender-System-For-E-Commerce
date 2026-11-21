"""Central place for paths and settings."""
import os

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    artifacts: Path = ROOT / "artifacts"

    def make(self) -> None:
        for p in (self.raw, self.processed, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)


#@dataclass(frozen=True)
#class DataConfig:
#    category: str = "Video_Games"           # Amazon Reviews 2023 category
#    max_review_rows = int = 1_500_000       # cap rows so it runs on a laptop
#    min_user_interactions: int = 5          # 5-core filter
#    min_item_interactions: int = 5
#    positive_rating_threshold: float = 4.0  # rating >= 4 => "liked it"
#    train_frac: float = 0.8
#    val_frac: float = 0.1                   # remaining 0.1 is test
#    seed: int = 42

@dataclass(frozen=True)
class DataConfig:
    category: str = os.environ.get("RECSYS_CATEGORY", "Video_Games")   # Amazon Reviews 2023 category
    max_review_rows: int = int(os.environ.get("RECSYS_MAX_ROWS", 1_500_000))   # 0 = read everything
    min_user_interactions: int = int(os.environ.get("RECSYS_MIN_USER", 5))
    min_item_interactions: int = int(os.environ.get("RECSYS_MIN_ITEM", 5))
    positive_rating_threshold: float = 4.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42

PATHS = Paths()
DATA = DataConfig()


BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEWS_URL = f"{BASE_URL}/review_categories/{DATA.category}.jsonl.gz"
META_URL = f"{BASE_URL}/meta_categories/meta_{DATA.category}.jsonl.gz"
    
