"""
stats.py
--------
Statistiques descriptives pour l'exploration du dataset Titanic.

Fonctions exposées :
- basic_overview(df) : infos clés (taille, valeurs manquantes, taux de survie)
- numeric_describe(df) : describe() des colonnes numériques, transposé
"""

from __future__ import annotations
import pandas as pd


def basic_overview(df: pd.DataFrame) -> dict:
    """
    Renvoie un dictionnaire de statistiques simples à afficher dans l'UI.

    - rows / cols
    - missing_total
    - survival_rate (si la colonne Survived existe)
    """
    overview = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
    }

    if "Survived" in df.columns and df.shape[0] > 0:
        # Survived est 0/1, donc la moyenne = taux de survie
        overview["survival_rate"] = float(df["Survived"].mean())

    return overview


def numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Décrit uniquement les colonnes numériques (plus lisible),
    et renvoie le tableau transposé (lignes = variables).
    """
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        return pd.DataFrame({"info": ["No numeric columns to describe."]})

    return num.describe().T
