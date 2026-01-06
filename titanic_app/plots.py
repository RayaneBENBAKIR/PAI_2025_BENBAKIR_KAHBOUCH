"""
plots.py
--------
Graphiques matplotlib pour l'app Streamlit.
On renvoie des figures (fig) et on ne fait jamais plt.show() ici.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def fig_age_hist_survivors(df: pd.DataFrame):
    """
    Histogramme des âges des survivants (Survived == 1).
    """
    fig, ax = plt.subplots()

    if "Survived" not in df.columns or "Age" not in df.columns:
        ax.text(0.5, 0.5, "Colonnes 'Survived'/'Age' manquantes", ha="center", va="center")
        ax.set_axis_off()
        return fig

    survivors_age = df[df["Survived"] == 1]["Age"].dropna()
    if survivors_age.empty:
        ax.text(0.5, 0.5, "Aucun survivant avec les filtres actuels", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.hist(survivors_age, bins=20)
    ax.set_title("Distribution des âges des survivants")
    ax.set_xlabel("Âge")
    ax.set_ylabel("Nombre de passagers")

    return fig


def fig_survivors_pie_by_sex(df: pd.DataFrame):
    """
    Camembert des survivants par sexe.
    """
    fig, ax = plt.subplots()

    if "Survived" not in df.columns or "Sex" not in df.columns:
        ax.text(0.5, 0.5, "Colonnes 'Survived'/'Sex' manquantes", ha="center", va="center")
        ax.set_axis_off()
        return fig

    survivors = df[df["Survived"] == 1]["Sex"].value_counts()
    if survivors.empty:
        ax.text(0.5, 0.5, "Aucun survivant avec les filtres actuels", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.pie(
        survivors.values,
        labels=survivors.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Répartition des survivants par sexe")

    return fig


def fig_survival_rate_by_class(df: pd.DataFrame):
    """
    Bar chart : taux de survie moyen par classe (Pclass).
    """
    fig, ax = plt.subplots()

    if "Survived" not in df.columns or "Pclass" not in df.columns:
        ax.text(0.5, 0.5, "Colonnes 'Survived'/'Pclass' manquantes", ha="center", va="center")
        ax.set_axis_off()
        return fig

    rates = df.groupby("Pclass")["Survived"].mean().sort_index()
    if rates.empty:
        ax.text(0.5, 0.5, "Pas de données après filtrage", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.bar(rates.index.astype(str), rates.values)
    ax.set_title("Taux de survie par classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Taux de survie")
    ax.set_ylim(0, 1)

    return fig
