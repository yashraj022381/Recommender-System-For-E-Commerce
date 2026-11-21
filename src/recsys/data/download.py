"""Download the Amazon Reviews 2023 files (reviews + product metadata)."""
import requests


from pathlib import Path
from tqdm import tqdm

from recsys.config import META_URL, PATHS, REVIEWS_URL



def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if tmp.stat().st_size < 1024:
        tmp.unlink()
        raise RuntimeError(f"Downloaded file from {url} is tiny; the URL is probably wrong.")
    tmp.replace(dest)                      # only a complete download gets the real name
    return dest



def download_all() -> tuple[Path, Path]:
    PATHS.make()
    reviews = _download(REVIEWS_URL, PATHS.raw / "reviews.jsonl.gz")
    meta = _download(META_URL, PATHS.raw / "meta.jsonl.gz")
    return reviews, meta


if __name__ == "__main__":
    download_all()



            
