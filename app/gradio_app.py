from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import DEFAULT_MODELS_DIR
from src.inference.predict import classify_mineral, load_artifacts

MODELS_DIR = PROJECT_ROOT / DEFAULT_MODELS_DIR
load_error = ""
try:
    models, label_encoder = load_artifacts(MODELS_DIR)
except Exception as exc:
    models, label_encoder = {}, None
    load_error = str(exc)

default_model = next(iter(models.keys())) if models else "Unavailable"


def _predict(image, model_name):
    if load_error:
        return load_error, {}
    try:
        prediction, confidence = classify_mineral(
            image=image,
            model_name=model_name,
            models=models,
            label_encoder=label_encoder,
        )
        return prediction, confidence
    except Exception as exc:
        return str(exc), {}


def build_interface() -> gr.Interface:
    return gr.Interface(
        fn=_predict,
        inputs=[
            gr.Image(type="pil", label="Upload Mineral Image"),
            gr.Dropdown(
                choices=list(models.keys()) if models else ["Unavailable"],
                value=default_model,
                label="Model",
            ),
        ],
        outputs=[
            gr.Textbox(label="Predicted Mineral"),
            gr.Label(label="Class Confidence"),
        ],
        title="Mineral Image Classifier",
        description="Classify mineral images using trained traditional ML models.",
    )


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(share=False)
