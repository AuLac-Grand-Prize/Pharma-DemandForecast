from __future__ import annotations

import logging
import os
import subprocess
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

M5_FILES = [
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
    "sample_submission.csv",
]


def _check_credentials() -> None:
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_json = Path.home().joinpath(".kaggle", "kaggle.json").exists()
    if not has_env and not has_json:
        raise RuntimeError(
            "Kaggle credentials not found. Either set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables, or place your API token at ~/.kaggle/kaggle.json."
        )


def download_m5(dest_dir: Path = Path("data/raw")) -> None:
    """Download M5 competition files via the Kaggle CLI into dest_dir.

    Idempotent: skips files that already exist. Requires Kaggle credentials.
    """
    _check_credentials()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check which files are already present
    missing = [f for f in M5_FILES if not (dest_dir / f).exists()]
    if not missing:
        logger.info("All M5 files already present in %s", dest_dir)
        return

    zip_path = dest_dir / "m5-forecasting-accuracy.zip"
    if not zip_path.exists():
        logger.info("Downloading M5 dataset via Kaggle CLI …")
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "m5-forecasting-accuracy",
                "-p",
                str(dest_dir),
            ],
            check=True,
        )

    logger.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for fname in M5_FILES:
            if not (dest_dir / fname).exists():
                logger.info("  Extracting %s", fname)
                zf.extract(fname, path=dest_dir)

    for fname in M5_FILES:
        fpath = dest_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            logger.info("  %s  %.1f MB", fname, size_mb)
        else:
            logger.warning("  %s not found after extraction", fname)

    logger.info("M5 download complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_m5()
