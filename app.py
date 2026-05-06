"""
Data Cleaning Agent

Run instructions:
pip install -r requirements.txt
streamlit run app.py
"""

import streamlit as st


def main() -> None:
    """Render the initial Streamlit interface."""
    st.set_page_config(page_title="Data Cleaning Agent", layout="wide")
    st.title("Data Cleaning Agent")
    st.write(
        "This starter app will load datasets, validate them, profile quality, "
        "clean records, and generate reports."
    )

    st.info("Project scaffold created. Build the workflow inside the utils modules.")


if __name__ == "__main__":
    main()
