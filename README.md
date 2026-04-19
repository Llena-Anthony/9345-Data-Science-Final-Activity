# Mineral Classification using Real and Synthetic Data

## Project Overview
This project evaluates machine learning models for mineral classification using real measured data and synthetically generated data. It focuses on handling class imbalance and determining whether synthetic data can improve performance on real-world samples.

## Objectives
- Establish baseline performance using real measured data
- Evaluate generalization of models trained on synthetic data
- Assess the impact of combining real and synthetic data
- Analyze performance under different class imbalance thresholds

## Dataset
- Real Data: Measured mineral dataset
- Synthetic Data: Generated mineral dataset
- Target Variable: mineral_name
- Features: Numerical mineral composition values (e.g., SiO2, Al2O3)

## Project Structure
project-root/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│       ├── thr10/
│       ├── thr20/
│       └── thr50/
├── src/
│   ├── models/
│   └── experiments/
│       ├── ex1_real.py
│       ├── ex2_synth_to_real.py
│       └── ex3_hybrid.py
├── results/
└── README.md

## Experiments

### Experiment 1: Real Baseline
- Train on real measured training data
- Test on real measured test data
- No synthetic data used

### Experiment 2: Synthetic to Real
- Train on synthetic data only
- Test on real measured test data
- Evaluation limited to shared classes

### Experiment 3: Hybrid
- Train on real measured training data combined with synthetic data
- Test on real measured test data
- Synthetic data used only for training

## Models
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

## Evaluation Metrics
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

## Thresholds
Class filtering is applied to the measured dataset using:
- thr10
- thr20
- thr50

Each threshold represents a minimum number of samples per class.

## Usage
Run each experiment separately:

python src/experiments/ex1_real.py
python src/experiments/ex2_synth_to_real.py
python src/experiments/ex3_hybrid.py

## Notes
- The test set is never used during training
- Synthetic data is never added to the test set
- Evaluation is restricted to valid classes only
- Train-test split is performed before any augmentation
