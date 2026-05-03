from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.config.settings import RANDOM_STATE


def fit_pca(
    X_train: np.ndarray,
    n_components: int | float | None = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_reduced = pca.fit_transform(X_train)
    return X_train_reduced, pca


def transform_with_pca(X: np.ndarray, pca: PCA) -> np.ndarray:
    return pca.transform(X)

