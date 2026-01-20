"""
Streamlit application: Titanic dataset explorer and survival prediction.

Run (from project root):
    python -m streamlit run titanic_app/app.py
"""

from __future__ import annotations

from pathlib import Path
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
from titanic_app.model import train_model, predict_survival


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def get_data():
    paths = DataPaths(PROJECT_ROOT)
    df_raw = load_titanic(paths.data_csv)
    df = prepare_titanic(df_raw)
    return df_raw, df


@st.cache_resource
def get_model(df_for_training):
    return train_model(df_for_training)


def sidebar_filters(df):
    st.sidebar.header("Filters")

    survived = st.sidebar.selectbox("Survival", ["All", "Survived", "Not Survived"])

    sex = "All"
    if "Sex" in df.columns:
        sex = st.sidebar.selectbox(
            "Sex",
            ["All"] + sorted(df["Sex"].dropna().unique().tolist()),
        )

    age_range = None
    if "Age" in df.columns and df["Age"].notna().any():
        age_range = st.sidebar.slider(
            "Age",
            min_value=float(df["Age"].min()),
            max_value=float(df["Age"].max()),
            value=(float(df["Age"].min()), float(df["Age"].max())),
        )

    pclass = None
    if "Pclass" in df.columns:
        classes = sorted(df["Pclass"].dropna().astype(int).unique().tolist())
        pclass = st.sidebar.multiselect("Class (Pclass)", classes, default=classes)

    embarked = "All"
    if "Embarked" in df.columns:
        ports = sorted(df["Embarked"].dropna().unique().tolist())
        embarked = st.sidebar.selectbox("Embarked", ["All"] + ports)

    alone = "All"
    if "Alone" in df.columns:
        alone = st.sidebar.selectbox("Travelling alone", ["All", "Yes", "No"])

    return survived, sex, age_range, pclass, embarked, alone


def render_explore_tab(df_filtered):
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Overview")
        st.write(basic_overview(df_filtered))

        st.subheader("Filtered data")
        st.dataframe(df_filtered, use_container_width=True, height=360)

        st.download_button(
            "Download filtered CSV",
            data=df_to_csv_bytes(df_filtered),
            file_name="titanic_filtered.csv",
            mime="text/csv",
        )

    with col_right:
        st.subheader("Numeric statistics")
        st.dataframe(numeric_describe(df_filtered), use_container_width=True, height=250)

        st.subheader("Plots")

        fig1 = fig_age_hist_survivors(df_filtered)
        st.pyplot(fig1)
        st.download_button(
            "Download plot (Age) as PNG",
            data=fig_to_png_bytes(fig1),
            file_name="plot_age.png",
            mime="image/png",
        )

        fig2 = fig_survivors_pie_by_sex(df_filtered)
        st.pyplot(fig2)
        st.download_button(
            "Download plot (Sex) as PNG",
            data=fig_to_png_bytes(fig2),
            file_name="plot_sex.png",
            mime="image/png",
        )

        fig3 = fig_survival_rate_by_class(df_filtered)
        st.pyplot(fig3)
        st.download_button(
            "Download plot (Class) as PNG",
            data=fig_to_png_bytes(fig3),
            file_name="plot_class.png",
            mime="image/png",
        )

        st.caption(
            "Pclass is the passenger socio-economic class: 1 = first class, "
            "2 = second class, 3 = third class."
        )


def render_prediction_tab(model):
    st.subheader("Survival prediction")

    with st.form("predict_form"):
        sex_in = st.selectbox("Sex", ["male", "female"])
        pclass_in = st.selectbox("Class (Pclass)", [1, 2, 3])
        age_in = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        fare_in = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0, step=1.0)
        sibsp_in = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
        parch_in = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        st.info("Fill the form and click Predict to run the model.")
        return

    features = {
        "Pclass": int(pclass_in),
        "Sex": sex_in,
        "Age": float(age_in),
        "Fare": float(fare_in),
        "SibSp": int(sibsp_in),
        "Parch": int(parch_in),
    }

    pred, proba = predict_survival(model, features)

    st.metric("Survival probability", f"{proba * 100:.1f}%")
    st.write(f"Predicted class: {pred} (1 = survived, 0 = not survived)")

    st.caption(
        "This model is trained on the Titanic dataset. "
        "It provides an indicative estimate, not a factual outcome."
    )


def main():
    st.set_page_config(page_title="Titanic Explorer", layout="wide")
    st.title("Titanic dataset explorer")

    try:
        _df_raw, df = get_data()
    except Exception as e:
        st.error("Failed to load data.")
        st.exception(e)
        st.stop()

    try:
        model = get_model(df)
    except Exception as e:
        st.error("Failed to train the prediction model.")
        st.exception(e)
        st.stop()

    survived, sex, age_range, pclass, embarked, alone = sidebar_filters(df)

    df_filtered = apply_filters(
        df,
        survived=survived,
        sex=sex,
        age_range=age_range,
        pclass=pclass,
        embarked=embarked,
        alone=alone,
    )

    tab_explore, tab_predict = st.tabs(["Explore", "Prediction"])

    with tab_explore:
        render_explore_tab(df_filtered)

    with tab_predict:
        render_prediction_tab(model)


if __name__ == "__main__":
    main()
