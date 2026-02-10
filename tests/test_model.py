import pandas as pd

from titanic_app.data_loader import prepare_titanic
from titanic_app.model import train_model, predict_survival


def test_train_model_returns_pipeline(sample_df: pd.DataFrame):
    """
    train_model should return a fitted sklearn Pipeline with predict_proba.
    """
    df = prepare_titanic(sample_df)
    model = train_model(df)

    assert hasattr(model, "predict_proba")


def test_predict_survival_returns_class_and_probability(sample_df: pd.DataFrame):
    """
    predict_survival should return:
    - a class in {0,1}
    - a probability between 0 and 1
    """
    df = prepare_titanic(sample_df)
    model = train_model(df)

    features = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "Fare": 7.25,
        "SibSp": 1,
        "Parch": 0,
    }

    pred, proba = predict_survival(model, features)

    assert pred in (0, 1)
    assert 0.0 <= proba <= 1.0
