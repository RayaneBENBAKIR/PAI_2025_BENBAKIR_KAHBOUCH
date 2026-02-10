import pandas as pd

from titanic_app.filters import apply_filters
from titanic_app.data_loader import prepare_titanic


def test_apply_filters_survival(sample_df: pd.DataFrame):
    df = prepare_titanic(sample_df)

    survived_only = apply_filters(df, survived="Survived")
    assert (survived_only["Survived"] == 1).all()

    not_survived_only = apply_filters(df, survived="Not Survived")
    assert (not_survived_only["Survived"] == 0).all()


def test_apply_filters_sex_and_class(sample_df: pd.DataFrame):
    df = prepare_titanic(sample_df)

    out = apply_filters(df, sex="female", pclass=[1])
    assert (out["Sex"] == "female").all()
    assert (out["Pclass"] == 1).all()
    assert out.shape[0] == 1


def test_apply_filters_embarked_and_alone(sample_df: pd.DataFrame):
    df = prepare_titanic(sample_df)

    out = apply_filters(df, embarked="S")
    assert (out["Embarked"] == "S").all()

    alone_yes = apply_filters(df, alone="Yes")
    assert (alone_yes["Alone"] == True).all()

    alone_no = apply_filters(df, alone="No")
    assert (alone_no["Alone"] == False).all()
