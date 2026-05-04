from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    IMAGE_INDEX_FILENAME,
    STRUCTURED_FEATURES_FILENAME,
)


def run_training_pipeline(
    dataset_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    results_dir: Path,
    overwrite_features: bool = False,
    n_pca_components: int | float | None = 0.95,
) -> None:
    from src.data.inspection import build_image_index
    from src.dimensionality.pca import fit_pca, transform_with_pca
    from src.evaluation.metrics import evaluate_models
    from src.features.pixel_features import create_or_load_structured_csv
    from src.features.scaling import fit_transform_scaler, transform_with_scaler
    from src.preprocessing.label_encoding import (
        encode_labels,
        split_features_and_labels,
        train_test_split_stratified,
    )
    from src.training.imbalance import apply_smote
    from src.training.train_models import get_default_models, train_models
    from src.utils.io import save_model_artifacts

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    index_csv = processed_dir / IMAGE_INDEX_FILENAME
    features_csv = processed_dir / STRUCTURED_FEATURES_FILENAME

    index_df = build_image_index(dataset_dir)
    index_df.to_csv(index_csv, index=False)

    feature_df = create_or_load_structured_csv(
        index_df=index_df,
        csv_path=features_csv,
        overwrite=overwrite_features,
    )

    X, label_series = split_features_and_labels(feature_df)
    y, label_encoder = encode_labels(label_series)

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Apply scaler
    X_train_scaled, scaler = fit_transform_scaler(X_train_bal)
    X_test_scaled = transform_with_scaler(X_test, scaler)

    # Apply PCA
    X_train_reduced, pca = fit_pca(X_train_scaled, n_components=n_pca_components)
    X_test_reduced = transform_with_pca(X_test_scaled, pca)

    n_components_kept = pca.n_components_
    variance_explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: reduced {X_train_scaled.shape[1]} features -> {n_components_kept} components "
        f"({variance_explained:.1%} variance retained)"
    )

    models = get_default_models()
    trained_models = train_models(models, X_train_reduced, y_train_bal)
    save_model_artifacts(trained_models, label_encoder, models_dir, scaler, pca)

    summary_df = evaluate_models(
        models=trained_models,
        X_test=X_test_reduced,
        y_test=y_test,
        label_encoder=label_encoder,
        results_dir=results_dir,
    )
    print(summary_df.sort_values("accuracy", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mineral image classification pipeline runner."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing class subfolders with mineral images.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for trained model artifacts.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Regenerate structured features CSV even if it exists.",
    )
    parser.add_argument(
        "--pca-components",
        type=float,
        default=0.95,
        help="PCA components to keep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_components: int | float = (
        int(args.pca_components)
        if args.pca_components >= 1
        else args.pca_components
    )
    run_training_pipeline(
        dataset_dir=args.dataset_dir,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        overwrite_features=args.overwrite_features,
        n_pca_components=n_components,
    )
