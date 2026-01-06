"""
export.py
---------
Fonctions utilitaires pour exporter :
- un DataFrame en CSV (bytes) pour download Streamlit
- une figure matplotlib en PNG (bytes) pour download Streamlit
"""

from __future__ import annotations

from io import BytesIO
import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convertit un DataFrame en CSV encodé en UTF-8 (bytes),
    utilisable directement dans st.download_button.
    """
    return df.to_csv(index=False).encode("utf-8")


def fig_to_png_bytes(fig) -> bytes:
    """
    Convertit une figure matplotlib en PNG (bytes),
    utilisable directement dans st.download_button.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    return buf.getvalue()
