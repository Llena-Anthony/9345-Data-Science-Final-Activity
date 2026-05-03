from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.settings import RANDOM_STATE, TEST_SIZE


def split_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    if "label" not in df.columns:
        raise KeyError("Expected 'label' column in feature dataframe.")
    X = df.drop(columns=["label"]).values
    labels = df["label"]
    return X, labels


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels.values)
    return y, label_encoder


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

