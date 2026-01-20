"""
model.py
--------
Simple survival prediction model for the Titanic dataset.

Model:
- Logistic Regression (scikit-learn)

Features:
- Pclass
- Sex
- Age
- Fare
- SibSp
- Parch
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


FEATURES = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]
TARGET = "Survived"


def train_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a logistic regression model and return a fitted sklearn Pipeline.
    """
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for training: {missing}")

    data = df[required].copy()

    # Simple imputations for robustness
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    categorical = ["Sex"]
    numeric = ["Pclass", "Age", "Fare", "SibSp", "Parch"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def predict_survival(model: Pipeline, features: dict) -> tuple[int, float]:
    """
    Predict survival (0/1) and survival probability.
    """
    X = pd.DataFrame([features], columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return pred, proba
