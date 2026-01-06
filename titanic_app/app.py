"""
Titanic Dataset Explorer (Streamlit)

Fonctionnalités :
- Chargement + préparation des données Titanic
- Filtres interactifs (survie, sexe, âge, classe)
- Statistiques descriptives
- Graphiques (histogramme, camembert, bar chart)
- Export CSV + export PNG

Lancement (depuis la racine du projet) :
    python -m streamlit run titanic_app/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

from titanic_app.data_loader import DataPaths, load_titanic, prepare_titanic
from titanic_app.filters import apply_filters
from titanic_app.stats import basic_overview, numeric_describe
from titanic_app.plots import (
    fig_age_hist_survivors,
    fig_survivors_pie_by_sex,
    fig_survival_rate_by_class,
)
from titanic_app.export import df_to_csv_bytes, fig_to_png_bytes


# =========================
# Configuration projet
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def get_data():
    """
    Charge les données depuis ./data/Titanic-Dataset.csv
    puis applique une préparation minimale.
    Cache Streamlit pour éviter de recharger à chaque interaction.
    """
    paths = DataPaths(PROJECT_ROOT)
    df_raw = load_titanic(paths.data_csv)
    df = prepare_titanic(df_raw)
    return df_raw, df


def sidebar_filters(df):
    """
    Construit la sidebar et renvoie les paramètres de filtres choisis.
    """
    st.sidebar.header("Filtres")

    survived = st.sidebar.selectbox("Survie", ["All", "Survived", "Not Survived"])

    sex = "All"
    if "Sex" in df.columns:
        sex = st.sidebar.selectbox(
            "Sexe",
            ["All"] + sorted(df["Sex"].dropna().unique().tolist()),
        )

    age_range = None
    if "Age" in df.columns and df["Age"].notna().any():
        age_range = st.sidebar.slider(
            "Âge",
            min_value=float(df["Age"].min()),
            max_value=float(df["Age"].max()),
            value=(float(df["Age"].min()), float(df["Age"].max())),
        )

    pclass = None
    if "Pclass" in df.columns:
        classes = sorted(df["Pclass"].dropna().astype(int).unique().tolist())
        pclass = st.sidebar.multiselect("Classe", classes, default=classes)

    return survived, sex, age_range, pclass


def render_left_column(df_filtered):
    """
    Colonne gauche : aperçu + table + export CSV.
    """
    st.subheader("📊 Aperçu")
    st.write(basic_overview(df_filtered))

    st.subheader("📋 Données filtrées")
    st.dataframe(df_filtered, use_container_width=True, height=360)

    st.download_button(
        "⬇️ Télécharger les données filtrées (CSV)",
        data=df_to_csv_bytes(df_filtered),
        file_name="titanic_filtered.csv",
        mime="text/csv",
    )


def render_right_column(df_filtered):
    """
    Colonne droite : stats + graphiques + exports PNG.
    """
    st.subheader("📈 Statistiques numériques")
    st.dataframe(numeric_describe(df_filtered), use_container_width=True, height=250)

    st.subheader("📉 Graphiques")

    fig1 = fig_age_hist_survivors(df_filtered)
    st.pyplot(fig1)
    st.download_button(
        "⬇️ Histogramme âges (PNG)",
        data=fig_to_png_bytes(fig1),
        file_name="age_hist_survivors.png",
        mime="image/png",
    )

    fig2 = fig_survivors_pie_by_sex(df_filtered)
    st.pyplot(fig2)
    st.download_button(
        "⬇️ Survivants par sexe (PNG)",
        data=fig_to_png_bytes(fig2),
        file_name="survivors_by_sex.png",
        mime="image/png",
    )

    fig3 = fig_survival_rate_by_class(df_filtered)
    st.pyplot(fig3)
    st.download_button(
        "⬇️ Survie par classe (PNG)",
        data=fig_to_png_bytes(fig3),
        file_name="survival_rate_by_class.png",
        mime="image/png",
    )


def main():
    """
    Point d'entrée Streamlit.
    """
    st.set_page_config(page_title="Titanic Dataset Explorer", layout="wide")

    # Debug utile (tu pourras l'enlever plus tard)
    st.write("✅ titanic_app/app.py chargé")
    st.write("📌 Python utilisé :", sys.executable)

    st.title("🚢 Titanic Dataset Explorer")
    st.caption("Exploration interactive du dataset Titanic (filtres, stats, graphes, export).")

    # Chargement sécurisé (affiche clairement les erreurs)
    try:
        _df_raw, df = get_data()
        st.success("✅ Données chargées")
    except Exception as e:
        st.error("❌ Erreur lors du chargement des données")
        st.exception(e)
        st.stop()

    # Filtres
    survived, sex, age_range, pclass = sidebar_filters(df)

    # Application des filtres
    df_filtered = apply_filters(
        df,
        survived=survived,
        sex=sex,
        age_range=age_range,
        pclass=pclass,
    )

    # Affichage
    col1, col2 = st.columns([1, 1])
    with col1:
        render_left_column(df_filtered)

    with col2:
        render_right_column(df_filtered)


if __name__ == "__main__":
    main()
