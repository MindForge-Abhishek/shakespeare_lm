"""Download the Tiny Shakespeare dataset into data/raw/."""

import requests
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "input.txt"

def download_dataset(url: str, output_path: Path) -> None:
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")

    print(f"Saved {len(response.text):,} characters to {output_path}")

if __name__ == "__main__":
    download_dataset(URL, OUTPUT_PATH)