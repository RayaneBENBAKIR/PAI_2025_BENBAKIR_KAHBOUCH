from pathlib import Path

import pandas as pd

from titanic_app.data_loader import load_titanic, prepare_titanic


def test_load_titanic_reads_csv(tmp_path: Path):
    """
    load_titanic should read a CSV file and return a DataFrame.
    We use a temporary file to avoid relying on project data files.
    """
    csv_path = tmp_path / "Titanic-Dataset.csv"
    csv_path.write_text("Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked\n1,1,female,29,0,0,71.2833,C\n")

    df = load_titanic(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert set(["Survived", "Pclass", "Sex"]).issubset(df.columns)


def test_prepare_titanic_fills_age_and_creates_alone(sample_df: pd.DataFrame):
    """
    prepare_titanic should:
    - fill missing Age with the median
    - create Alone = (SibSp + Parch == 0)
    - drop PassengerId, Ticket, Cabin (if present)
    - keep Embarked
    """
    out = prepare_titanic(sample_df)

    assert "Age" in out.columns
    assert out["Age"].isna().sum() == 0

    assert "Alone" in out.columns
    # row 0 has SibSp=1, Parch=0 -> not alone
    assert bool(out.loc[0, "Alone"]) is False
    # row 1 has SibSp=0, Parch=0 -> alone
    assert bool(out.loc[1, "Alone"]) is True

    
    


    assert "PassengerId" not in out.columns
    assert "Ticket" not in out.columns
    assert "Cabin" not in out.columns

    assert "Embarked" in out.columns
