import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Small in-memory Titanic-like dataset for testing.

    Notes:
    - Includes missing Age and missing Embarked to validate preparation behavior.
    - Keeps only columns we actually use in the project.
    """
    return pd.DataFrame(
        {
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Sex": ["male", "female", "female", "male"],
            "Age": [22.0, None, 14.0, 35.0],
            "SibSp": [1, 0, 1, 0],
            "Parch": [0, 0, 2, 0],
            "Fare": [7.25, 71.2833, 7.925, 13.0],
            "Embarked": ["S", "C", "S", None],
            "PassengerId": [1, 2, 3, 4],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803"],
            "Cabin": [None, "C85", None, None],
        }
    )
