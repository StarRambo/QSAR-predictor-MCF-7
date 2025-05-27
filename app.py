import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("QSAR Model: pIC50 Predictor (No RDKit)")

# Load model
model = joblib.load("rf_model.pkl")

# File upload
uploaded_file = st.file_uploader("Upload CSV with 1022-bit ECFP4 fingerprints (Bit_0 to Bit_1021)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    bit_columns = [col for col in df.columns if col.startswith("Bit_")]

    if len(bit_columns) != 1022:
        st.error("Expected 1022 fingerprint bits.")
    else:
        X = df[bit_columns].values
        predictions = model.predict(X)
        df["Predicted_pIC50"] = predictions

        st.success("Prediction complete!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
