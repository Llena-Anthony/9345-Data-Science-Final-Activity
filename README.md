# Performance Evaluation of Traditional Machine Learning Models Using High-Dimensional Pixel-Based Features for Mineral Image Classification

## Project Overview
This repository evaluates traditional machine learning models for classifying mineral images using flattened RGB pixel features extracted from resized images.

## Dataset
- Source notebook references: `youcefattallah97/minerals-identification-classification` (Kaggle)
- Expected local structure for training:
  - `data/raw/Minet 5640 Images/<class_name>/*.jpg|*.png|...`

Raw and processed data folders are included as placeholders and are intentionally ignored for Git tracking.

## Repository Structure
- `app/gradio_app.py`: Gradio inference interface
- `src/`: modular training and inference code
- `notebooks/main_experiment.ipynb`: migrated experiment notebook
- `docs/`: pipeline and model selection notes
- `models/`: trained model artifacts
- `results/`: evaluation outputs (reports and confusion matrices)

## Workflow
1. Build image index from class-based folders.
2. Convert images to flattened `64x64 RGB` feature vectors.
3. Encode labels and perform stratified train/test split.
4. Apply SMOTE to training data.
5. Train Random Forest, Naive Bayes, KNN, and SVM.
6. Evaluate models (accuracy, classification report, confusion matrix).
7. Save model artifacts and evaluation outputs.
8. Run Gradio app for inference.

## Models Evaluated
- Random Forest
- Naive Bayes (Gaussian)
- K-Nearest Neighbors
- Support Vector Machine (RBF)

## Evaluation Outputs
Generated in `results/`:
- `model_accuracy_summary.csv`
- `<model>_classification_report.csv`
- `<model>_confusion_matrix.csv`

## Setup
```bash
pip install -r requirements.txt
```

## Run Training Pipeline
```bash
python main.py --dataset-dir "data/raw/Minet 5640 Images"
```

Optional flags:
- `--processed-dir data/processed`
- `--models-dir models`
- `--results-dir results`
- `--overwrite-features`

## Run Gradio App
```bash
python app/gradio_app.py
```

Note: the app requires trained model files in `models/` (`mineral_*_model.pkl`) and `label_encoder.pkl`.
