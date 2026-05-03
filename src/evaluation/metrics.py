from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.config.settings import MODEL_SUMMARY_FILENAME
from src.evaluation.visualization import (
    save_confusion_matrix_csv,
    save_confusion_matrix_plot,
)
from src.utils.paths import ensure_directory


def evaluate_models(
    models: dict[str, object],
    X_test,
    y_test,
    label_encoder: LabelEncoder,
    results_dir: str | Path | None = None,
) -> pd.DataFrame:
    class_names = label_encoder.classes_
    rows: list[dict[str, float | str]] = []
    save_dir = Path(results_dir) if results_dir is not None else None

    if save_dir is not None:
        ensure_directory(save_dir)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        rows.append({"model": name, "accuracy": float(accuracy)})

        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        cm_df = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=class_names,
            columns=class_names,
        )

        if save_dir is not None:
            safe_name = name.lower().replace(" ", "_")
            report_df.to_csv(save_dir / f"{safe_name}_classification_report.csv")
            save_confusion_matrix_csv(cm_df=cm_df, output_dir=save_dir, model_name=name)
            save_confusion_matrix_plot(cm_df=cm_df, output_dir=save_dir, model_name=name)

    summary_df = pd.DataFrame(rows)
    if save_dir is not None:
        summary_df.to_csv(save_dir / MODEL_SUMMARY_FILENAME, index=False)
    return summary_df

