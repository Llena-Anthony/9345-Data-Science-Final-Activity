from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config.settings import IMAGE_SIZE


def to_rgb_resized_array(
    image: Image.Image, size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    return np.array(image.convert("RGB").resize(size))


def load_rgb_resized_array(
    image_path: str | Path, size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    with Image.open(image_path) as image:
        return to_rgb_resized_array(image=image, size=size)

