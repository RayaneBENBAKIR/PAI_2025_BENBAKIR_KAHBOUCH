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
