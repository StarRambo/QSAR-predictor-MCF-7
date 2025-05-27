import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("QSAR Model: pIC50 Predictor (No RDKit Version)")

# Load trained model
rf_model = joblib.load("rf_model.pkl")

# Upload fingerprint CSV
uploaded_file = st.file_uploader("Upload CSV with precomputed ECFP4 (1022-bit) fingerprints", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Identify fingerprint columns
    bit_columns = [col for col in df.columns if col.startswith("Bit_")]

    if len(bit_columns) != 1022:
        st.error("Expected 1022 fingerprint bits (Bit_0 to Bit_1021).")
    else:
        X = df[bit_columns].values
        predictions = rf_model.predict(X)
        df["Predicted_pIC50"] = predictions

        st.success("Prediction complete!")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "qsar_predictions.csv", "text/csv")
