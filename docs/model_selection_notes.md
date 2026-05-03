# Model Selection Notes

## Why Traditional ML
The experiment uses flattened pixel vectors (`64 x 64 x 3 = 12,288` features) and compares conventional classifiers under the same feature representation.

## Candidate Models
- **Random Forest**
  - Handles nonlinear relationships.
  - Robust on mixed decision boundaries.
  - Provides a strong baseline for tabularized image pixels.
- **Naive Bayes (Gaussian)**
  - Fast to train and evaluate.
  - Useful as a probabilistic baseline despite independence assumptions.
- **K-Nearest Neighbors**
  - Non-parametric reference model.
  - Sensitive to feature scaling and local neighborhood structure.
- **Support Vector Machine (RBF)**
  - Effective with high-dimensional spaces.
  - More computationally expensive than other candidates.

## Evaluation Metrics
- Overall accuracy
- Precision, recall, and F1-score per class
- Confusion matrix per model

## Practical Notes
- SMOTE is applied only to the training split.
- Current Gradio app requires serialized model files in `models/`.
- If only `label_encoder.pkl` exists, retraining via `main.py` is required before app inference.
