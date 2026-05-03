from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_directory


def save_confusion_matrix_csv(
    cm_df: pd.DataFrame, output_dir: str | Path, model_name: str
) -> Path:
    output_dir = ensure_directory(output_dir)
    safe_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_name}_confusion_matrix.csv"
    cm_df.to_csv(output_path)
    return output_path


def save_confusion_matrix_plot(
    cm_df: pd.DataFrame, output_dir: str | Path, model_name: str
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_dir = ensure_directory(output_dir)
    safe_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_name}_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm_df.values, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} Confusion Matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

