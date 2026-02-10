![CI](https://github.com/RayaneBENBAKIR/PAI_2025_BENBAKIR_KAHBOUCH/actions/workflows/ci.yml/badge.svg)

# Titanic Survival Analysis and Prediction

This project explores the Titanic dataset through interactive data analysis and
a simple survival prediction model.

The application is built with Streamlit and is intended for educational purposes.

---

## Project objectives

- Explore the main socio-demographic factors influencing survival on the Titanic
- Provide clear visualizations to support data interpretation
- Offer an interactive survival prediction tool based on a machine learning model

---

## Dataset

The dataset used is the classic Titanic dataset, containing information about
passengers such as age, sex, socio-economic class, family relations, and survival outcome.

---

## Application features

### Data exploration

- Interactive filters (sex, age, class, embarkation port, travelling alone)
- Summary statistics and descriptive tables
- Visualizations:
  - Survival rate by age group
  - Survival rate by sex
  - Survival rate by socio-economic class

### Survival prediction

- Logistic regression model trained on the dataset
- Input features:
  - Pclass (socio-economic class)
  - Sex
  - Age
  - Fare
  - SibSp (siblings/spouses aboard)
  - Parch (parents/children aboard)
  - Output:
  - Survival probability
  - Binary prediction (survived / not survived)

---

## Project structure

.
├── titanic_app/
│ ├── app.py # Streamlit application entry point
│ ├── data_loader.py # Data loading and preprocessing
│ ├── filters.py # Dataset filtering utilities
│ ├── plots.py # Visualization functions
│ ├── model.py # Survival prediction model
│ └── export.py # Data and figure export helpers
├── tests/
│ ├── conftest.py # Shared pytest fixtures
│ ├── test_data_loader.py
│ ├── test_filters.py
│ ├── test_model.py
│ └── test_stats.py
├── docs/
│ └── source/ # Sphinx documentation sources
├── data/
│ └── Titanic-Dataset.csv
├── .github/
│ └── workflows/
│ └── ci.yml # GitHub Actions CI configuration
├── requirements.txt
├── pytest.ini
└── README.md
