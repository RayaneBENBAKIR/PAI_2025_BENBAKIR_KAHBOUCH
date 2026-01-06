"""
data_loader.py
--------------
Chargement et préparation du dataset Titanic.

- Charge le CSV depuis ./data/Titanic-Dataset.csv
- Prépare la data (imputation Age, drop colonnes inutiles)
- Pas de chemins absolus (portable + GitHub friendly)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    """Centralise les chemins utilisés dans le projet."""
    root: Path

    @property
    def data_csv(self) -> Path:
        # Chemin attendu : <root>/data/Titanic-Dataset.csv
        return self.root / "data" / "Titanic-Dataset.csv"


def load_titanic(csv_path: Path) -> pd.DataFrame:
    """Charge le dataset Titanic depuis un fichier CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable: {csv_path}\n"
            "➡️ Place Titanic-Dataset.csv dans le dossier ./data/ (à la racine du projet)."
        )
    return pd.read_csv(csv_path)


def prepare_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Préparation minimale inspirée de ton main.py :
    - Age : remplace les NaN par la médiane
    - Drop colonnes : PassengerId, Ticket, Cabin, Embarked (si présentes)
    """
    df2 = df.copy()

    if "Age" in df2.columns:
        df2["Age"] = df2["Age"].fillna(df2["Age"].median())

    cols_to_drop = [c for c in ["PassengerId", "Ticket", "Cabin", "Embarked"] if c in df2.columns]
    if cols_to_drop:
        df2 = df2.drop(columns=cols_to_drop)

    return df2
