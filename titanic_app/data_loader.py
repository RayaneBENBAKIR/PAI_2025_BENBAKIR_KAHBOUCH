"""
Data loading and basic preparation for the Titanic dataset.

Expected data location:
    <project_root>/data/Titanic-Dataset.csv

Preparation:
- Fill missing Age with the median
- Create a boolean column Alone = (SibSp + Parch == 0)
- Drop columns that are not useful for analysis (PassengerId, Ticket, Cabin)
- Keep Embarked (used as a filter)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def data_csv(self) -> Path:
        return self.root / "data" / "Titanic-Dataset.csv"


def load_titanic(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Place Titanic-Dataset.csv in <project_root>/data/."
        )
    return pd.read_csv(csv_path)


def prepare_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    if "Age" in df2.columns:
        df2["Age"] = df2["Age"].fillna(df2["Age"].median())

    if "SibSp" in df2.columns and "Parch" in df2.columns:
        df2["Alone"] = (df2["SibSp"] + df2["Parch"] == 0)

    cols_to_drop = [c for c in ["PassengerId", "Ticket", "Cabin"] if c in df2.columns]
    if cols_to_drop:
        df2 = df2.drop(columns=cols_to_drop)

    return df2
