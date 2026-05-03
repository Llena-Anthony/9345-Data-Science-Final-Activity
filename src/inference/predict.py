from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config.settings import IMAGE_SIZE
from src.preprocessing.image_preprocessing import to_rgb_resized_array
from src.utils.io import load_model_artifacts


def load_artifacts(models_dir: str | Path) -> tuple[dict[str, object], object]:
    return load_model_artifacts(models_dir=models_dir)


def classify_mineral(
    image: Image.Image,
    model_name: str,
    models: dict[str, object],
    label_encoder,
    size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[str, dict[str, float]]:
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not loaded.")

    features = to_rgb_resized_array(image=image, size=size).flatten().reshape(1, -1)
    model = models[model_name]
    pred_index = int(model.predict(features)[0])
    pred_label = str(label_encoder.inverse_transform([pred_index])[0])

    confidence: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        for idx, prob in enumerate(probs):
            label = str(label_encoder.inverse_transform([idx])[0])
            confidence[label] = float(prob)

    return pred_label, confidence

