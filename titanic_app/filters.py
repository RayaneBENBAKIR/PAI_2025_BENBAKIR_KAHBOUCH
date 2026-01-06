"""
filters.py
----------
Fonctions pures de filtrage (faciles à tester).

Filtres gérés :
- survived : "All" | "Survived" | "Not Survived"
- sex      : "All" | "male" | "female" (ou toute valeur présente)
- age_range: (min_age, max_age)
- pclass   : liste d'entiers (ex: [1,2,3])
"""

from __future__ import annotations
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    survived: str = "All",
    sex: str = "All",
    age_range: tuple[float, float] | None = None,
    pclass: list[int] | None = None,
) -> pd.DataFrame:
    """
    Applique les filtres à un DataFrame Titanic et retourne un nouveau DataFrame.

    Remarque :
    - On ne modifie jamais df en place (bonnes pratiques).
    """
    out = df.copy()

    # Filtre survie
    if survived != "All" and "Survived" in out.columns:
        if survived == "Survived":
            out = out[out["Survived"] == 1]
        elif survived == "Not Survived":
            out = out[out["Survived"] == 0]

    # Filtre sexe
    if sex != "All" and "Sex" in out.columns:
        out = out[out["Sex"] == sex]

    # Filtre âge
    if age_range is not None and "Age" in out.columns:
        a_min, a_max = age_range
        out = out[(out["Age"] >= a_min) & (out["Age"] <= a_max)]

    # Filtre classe
    if pclass is not None and len(pclass) > 0 and "Pclass" in out.columns:
        out = out[out["Pclass"].isin(pclass)]

    return out
