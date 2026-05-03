from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE

from src.config.settings import RANDOM_STATE


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    sampler = SMOTE(random_state=random_state)
    return sampler.fit_resample(X_train, y_train)

