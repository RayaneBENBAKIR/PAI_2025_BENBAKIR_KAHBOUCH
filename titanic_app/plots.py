"""
plots.py

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def _empty_fig(message: str):
    """Crée une figure vide avec un message centré (+robuste si filtres => df vide)."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _annotate_bars(ax, x, y, fmt="{:.0f}%"):
    """Ajoute des labels au-dessus des barres."""
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.02, fmt.format(yi * 100), ha="center", va="bottom", fontsize=10)


def fig_age_hist_survivors(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("Aucune donnée après filtrage.")
    if "Age" not in df.columns or "Survived" not in df.columns:
        return _empty_fig("Colonnes 'Age' / 'Survived' manquantes.")

    # On évite les NaN
    data = df[["Age", "Survived"]].dropna()
    if data.empty:
        return _empty_fig("Aucune donnée (Age/Survived) après filtrage.")

    # Bins d'âge (tu peux ajuster)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]

    data = data.copy()
    data["AgeBin"] = pd.cut(data["Age"], bins=bins, labels=labels, include_lowest=True, right=False)

    # Taux + effectif par tranche
    grouped = data.groupby("AgeBin", observed=True)["Survived"]
    rate = grouped.mean()
    count = grouped.size()

    # On garde seulement les bins non vides
    rate = rate[count > 0]
    count = count[count > 0]

    if rate.empty:
        return _empty_fig("Pas assez de données pour calculer les tranches d'âge.")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(rate)))
    ax.bar(x, rate.values)

    ax.set_title("Taux de survie par tranche d'âge")
    ax.set_xlabel("Tranche d'âge")
    ax.set_ylabel("Taux de survie")
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(rate.index.astype(str), rotation=45, ha="right")

    # Annotations : % + n
    _annotate_bars(ax, x, rate.values, fmt="{:.0f}%")
    for xi, n in zip(x, count.values):
        ax.text(xi, 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def fig_survivors_pie_by_sex(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("Aucune donnée après filtrage.")
    if "Sex" not in df.columns or "Survived" not in df.columns:
        return _empty_fig("Colonnes 'Sex' / 'Survived' manquantes.")

    data = df[["Sex", "Survived"]].dropna()
    if data.empty:
        return _empty_fig("Aucune donnée (Sex/Survived) après filtrage.")

    rate = data.groupby("Sex")["Survived"].mean().sort_index()
    count = data.groupby("Sex")["Survived"].size().reindex(rate.index)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(rate)))
    ax.bar(x, rate.values)

    ax.set_title("Taux de survie par sexe")
    ax.set_xlabel("Sexe")
    ax.set_ylabel("Taux de survie")
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels(rate.index.astype(str))

    _annotate_bars(ax, x, rate.values, fmt="{:.0f}%")

    # effectifs n=...
    for xi, n in zip(x, count.values):
        ax.text(xi, 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def fig_survival_rate_by_class(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("Aucune donnée après filtrage.")
    if "Pclass" not in df.columns or "Survived" not in df.columns:
        return _empty_fig("Colonnes 'Pclass' / 'Survived' manquantes.")

    data = df[["Pclass", "Survived"]].dropna()
    if data.empty:
        return _empty_fig("Aucune donnée (Pclass/Survived) après filtrage.")

    rate = data.groupby("Pclass")["Survived"].mean().sort_index()
    count = data.groupby("Pclass")["Survived"].size().reindex(rate.index)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(rate)))
    ax.bar(x, rate.values)

    ax.set_title("Taux de survie par classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Taux de survie")
    ax.set_ylim(0, 1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in rate.index.tolist()])

    _annotate_bars(ax, x, rate.values, fmt="{:.0f}%")
    for xi, n in zip(x, count.values):
        ax.text(xi, 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


# BONUS (pas utilisé dans app.py pour l'instant) :
def fig_heatmap_survival_sex_class(df: pd.DataFrame):
    if df.empty:
        return _empty_fig("Aucune donnée après filtrage.")
    if not {"Sex", "Pclass", "Survived"}.issubset(df.columns):
        return _empty_fig("Colonnes 'Sex', 'Pclass', 'Survived' manquantes.")

    pivot = (
        df.dropna(subset=["Sex", "Pclass", "Survived"])
        .pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean")
        .sort_index()
    )

    if pivot.empty:
        return _empty_fig("Pas assez de données pour la heatmap.")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Taux de survie (Sex × Classe)")
    ax.set_xlabel("Classe (Pclass)")
    ax.set_ylabel("Sexe")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns.tolist()])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())

    # valeurs dans la heatmap
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
