import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://data.deepai.org/text8.zip"

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
ARCHIVE   = DATA_DIR / "text8.zip"
DATASET    = DATA_DIR / "text8"


def fetch_dataset() -> None:
    if DATASET.exists():
        print(f"dataset already present at {DATASET}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"downloading text8 from {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, ARCHIVE)

    print("unpacking archive ...")
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        zf.extractall(DATA_DIR)

    ARCHIVE.unlink()  # remove zip, keep only the extracted text
    print(f"ready — dataset saved to {DATASET}")


if __name__ == "__main__":
    fetch_corpus()
