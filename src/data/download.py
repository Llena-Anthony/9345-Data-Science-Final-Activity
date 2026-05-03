from __future__ import annotations

from pathlib import Path

from src.config.settings import DATASET_KAGGLE_ID


def download_kaggle_dataset(dataset_id: str = DATASET_KAGGLE_ID) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for dataset download. Install with: pip install kagglehub"
        ) from exc

    downloaded_path = kagglehub.dataset_download(dataset_id)
    return Path(downloaded_path)

