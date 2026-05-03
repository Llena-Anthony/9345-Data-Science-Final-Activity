from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config.settings import LABEL_ENCODER_FILENAME, MODEL_FILENAMES
from src.utils.paths import ensure_directory


def save_pickle(obj, file_path: str | Path) -> None:
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    joblib.dump(obj, file_path)


def load_pickle(file_path: str | Path):
    return joblib.load(Path(file_path))


def save_dataframe_csv(df: pd.DataFrame, file_path: str | Path) -> None:
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    df.to_csv(file_path, index=False)


def save_model_artifacts(
    models: dict[str, object], label_encoder, output_dir: str | Path
) -> None:
    output_dir = ensure_directory(output_dir)
    for model_name, filename in MODEL_FILENAMES.items():
        if model_name in models:
            save_pickle(models[model_name], output_dir / filename)
    save_pickle(label_encoder, output_dir / LABEL_ENCODER_FILENAME)


def load_model_artifacts(
    models_dir: str | Path,
) -> tuple[dict[str, object], object]:
    models_dir = Path(models_dir)
    label_encoder_path = models_dir / LABEL_ENCODER_FILENAME
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder: {label_encoder_path}")

    label_encoder = load_pickle(label_encoder_path)
    models: dict[str, object] = {}
    for model_name, filename in MODEL_FILENAMES.items():
        model_path = models_dir / filename
        if model_path.exists():
            models[model_name] = load_pickle(model_path)

    if not models:
        raise FileNotFoundError(
            f"No model files found in {models_dir}. Expected one of: {list(MODEL_FILENAMES.values())}"
        )

    return models, label_encoder

