from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config.settings import RANDOM_STATE


def get_default_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


def train_models(
    models: dict[str, object], X_train: np.ndarray, y_train: np.ndarray
) -> dict[str, object]:
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} done.")
    return models

