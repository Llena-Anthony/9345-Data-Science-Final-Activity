from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import IMAGE_SIZE, PIXEL_FEATURE_COUNT
from src.preprocessing.image_preprocessing import load_rgb_resized_array
from src.utils.paths import ensure_directory


def image_to_vector(
    image_path: str | Path, size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    return load_rgb_resized_array(image_path=image_path, size=size).flatten()


def build_feature_table(
    index_df: pd.DataFrame, size: tuple[int, int] = IMAGE_SIZE
) -> pd.DataFrame:
    if not {"filepath", "label"}.issubset(index_df.columns):
        raise KeyError("Input dataframe must include 'filepath' and 'label'.")

    vectors = [image_to_vector(path, size=size) for path in index_df["filepath"]]
    feature_count = PIXEL_FEATURE_COUNT if size == IMAGE_SIZE else size[0] * size[1] * 3

    feature_df = pd.DataFrame(
        vectors, columns=[f"pixel_{i}" for i in range(feature_count)]
    )
    feature_df.insert(0, "label", index_df["label"].values)
    return feature_df


def create_or_load_structured_csv(
    index_df: pd.DataFrame,
    csv_path: str | Path,
    size: tuple[int, int] = IMAGE_SIZE,
    overwrite: bool = False,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if csv_path.exists() and not overwrite:
        return pd.read_csv(csv_path)

    feature_df = build_feature_table(index_df=index_df, size=size)
    ensure_directory(csv_path.parent)
    feature_df.to_csv(csv_path, index=False)
    return feature_df
