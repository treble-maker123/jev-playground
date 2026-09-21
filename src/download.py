"""Download a dataset into the local data/ folder.

Kaggle competitions require API credentials -- run `uv run kaggle auth login`,
or set KAGGLE_API_TOKEN from https://www.kaggle.com/settings/api. You also have
to accept the competition's rules on its website once.

Public Hugging Face datasets need no credentials. For gated or private repos,
run `uv run hf auth login` or set HF_TOKEN first.
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_kaggle(competition: str, name: str) -> Path:
    target = DATA_DIR / name
    target.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path=str(target), quiet=False)

    for archive in sorted(target.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        archive.unlink()

    return target


def download_hf(repo_id: str, name: str) -> Path:
    target = DATA_DIR / name
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target))

    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--competition",
        help="Kaggle competition slug to download (e.g. titanic).",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Hugging Face dataset repo id (e.g. nicoco404/AITA_labeled_posts).",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Folder name under data/ (default: the slug or repo name, lowercased).",
    )
    args = parser.parse_args()

    if bool(args.competition) == bool(args.dataset):
        parser.error("pass exactly one of --competition or --dataset")

    source = args.competition or args.dataset
    name = args.name or source.split("/")[-1].lower()

    if args.competition:
        target = download_kaggle(args.competition, name)
    else:
        target = download_hf(args.dataset, name)

    files = sorted(p.name for p in target.iterdir() if not p.name.startswith("."))
    print(f"Downloaded to {target}: {', '.join(files)}")


if __name__ == "__main__":
    main()
