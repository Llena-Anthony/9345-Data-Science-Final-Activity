# Project Pipeline

## Objective
Evaluate traditional machine learning models using high-dimensional pixel-based image features for mineral image classification.

## Workflow
1. Collect mineral images grouped by class in a folder structure under `data/raw/`.
2. Build an index of image file paths and class labels.
3. Resize each image to `64x64`, flatten RGB pixels, and store features in a structured CSV.
4. Encode class labels with `LabelEncoder`.
5. Split data into train/test sets with stratification.
6. Apply SMOTE on the training split to reduce class imbalance.
7. Train baseline models:
   - Random Forest
   - Naive Bayes
   - K-Nearest Neighbors
   - Support Vector Machine (RBF kernel)
8. Evaluate using accuracy, per-class metrics, and confusion matrices.
9. Save model artifacts and evaluation outputs.
10. Serve inference through Gradio.

## Folder Responsibility
- `src/`: reusable pipeline modules
- `main.py`: pipeline runner
- `app/gradio_app.py`: user-facing inference interface
- `notebooks/main_experiment.ipynb`: original exploratory workflow
- `results/`: evaluation outputs
- `models/`: serialized model artifacts
