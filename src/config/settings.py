from __future__ import annotations

from pathlib import Path

PROJECT_TITLE = (
    "Performance Evaluation of Traditional Machine Learning Models Using "
    "High-Dimensional Pixel-Based Features for Mineral Image Classification"
)

DATASET_KAGGLE_ID = "youcefattallah97/minerals-identification-classification"
DATASET_FOLDER_NAME = "Minet 5640 Images"

DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_PROCESSED_DATA_DIR = Path("data/processed")
DEFAULT_DATASET_DIR = DEFAULT_RAW_DATA_DIR / DATASET_FOLDER_NAME
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("results")

IMAGE_INDEX_FILENAME = "image_index.csv"
STRUCTURED_FEATURES_FILENAME = "minerals_structured.csv"
MODEL_SUMMARY_FILENAME = "model_accuracy_summary.csv"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

IMAGE_SIZE = (64, 64)
PIXEL_FEATURE_COUNT = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_FILENAMES = {
    "Random Forest": "mineral_rf_model.pkl",
    "Naive Bayes": "mineral_nb_model.pkl",
    "KNN": "mineral_knn_model.pkl",
    "SVM": "mineral_svm_model.pkl",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

