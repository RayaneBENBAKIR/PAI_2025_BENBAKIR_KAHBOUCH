"""
Filtering utilities.

Filters supported:
- survived: "All" | "Survived" | "Not Survived"
- sex: "All" or a value present in df["Sex"]
- age_range: (min_age, max_age)
- pclass: list of ints
- embarked: "All" or one of the ports (typically C, Q, S)
- alone: "All" | "Yes" | "No"
"""

from __future__ import annotations
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    survived: str = "All",
    sex: str = "All",
    age_range: tuple[float, float] | None = None,
    pclass: list[int] | None = None,
    embarked: str = "All",
    alone: str = "All",
) -> pd.DataFrame:
    out = df.copy()

    if survived != "All" and "Survived" in out.columns:
        if survived == "Survived":
            out = out[out["Survived"] == 1]
        elif survived == "Not Survived":
            out = out[out["Survived"] == 0]

    if sex != "All" and "Sex" in out.columns:
        out = out[out["Sex"] == sex]

    if age_range is not None and "Age" in out.columns:
        a_min, a_max = age_range
        out = out[(out["Age"] >= a_min) & (out["Age"] <= a_max)]

    if pclass is not None and len(pclass) > 0 and "Pclass" in out.columns:
        out = out[out["Pclass"].isin(pclass)]

    if embarked != "All" and "Embarked" in out.columns:
        out = out[out["Embarked"] == embarked]

    if alone != "All" and "Alone" in out.columns:
        if alone == "Yes":
            out = out[out["Alone"] == True]
        elif alone == "No":
            out = out[out["Alone"] == False]

    return out
