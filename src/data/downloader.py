"""Download and extract the MovieLens ``ml-latest-small`` dataset.

Usage:
    python -m src.data.downloader [--force]
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import zipfile
from pathlib import Path

import requests

from src.config import DATASET_FILES, DATASET_NAME, DATASET_URL, RAW_DATA_DIR

logger = logging.getLogger(__name__)


def _files_present(target_dir: Path) -> bool:
    return all((target_dir / name).is_file() for name in DATASET_FILES)


def download_movielens(
    target_dir: Path = RAW_DATA_DIR,
    url: str = DATASET_URL,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download the MovieLens archive and extract its CSV files into ``target_dir``.

    Args:
        target_dir: Directory where ``movies.csv``, ``ratings.csv``, ``tags.csv`` and
            ``links.csv`` are written.
        url: Location of the zip archive.
        force: Re-download even when all files already exist.
        timeout: HTTP timeout in seconds.

    Returns:
        The directory containing the extracted CSV files.

    Raises:
        requests.HTTPError: If the download fails.
        FileNotFoundError: If the archive does not contain the expected files.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if _files_present(target_dir) and not force:
        logger.info("Dataset already present in %s — skipping download.", target_dir)
        return target_dir

    logger.info("Downloading %s ...", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in DATASET_FILES:
            member = f"{DATASET_NAME}/{name}"
            if member not in archive.namelist():
                raise FileNotFoundError(f"'{member}' not found in downloaded archive.")
            with archive.open(member) as src, open(target_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)

    logger.info("Extracted %d files to %s", len(DATASET_FILES), target_dir)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the MovieLens small dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--target-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    download_movielens(target_dir=args.target_dir, force=args.force)


if __name__ == "__main__":
    main()
