from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_transform_scaler(X_train: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler


def transform_with_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)

