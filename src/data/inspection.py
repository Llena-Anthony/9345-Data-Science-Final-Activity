from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import SUPPORTED_IMAGE_EXTENSIONS


def build_image_index(dataset_dir: str | Path) -> pd.DataFrame:
    records = []
    dataset_dir = Path(dataset_dir)

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                records.append(
                    {"filepath": str(image_path.resolve()), "label": class_dir.name}
                )

    if not records:
        raise ValueError(f"No supported image files found under: {dataset_dir}")

    return pd.DataFrame(records)


def class_distribution(df: pd.DataFrame) -> pd.Series:
    if "label" not in df.columns:
        raise KeyError("Expected 'label' column in dataframe.")
    return df["label"].value_counts().sort_values(ascending=False)

