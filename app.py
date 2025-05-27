import streamlit as st
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

# Load model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Convert SMILES to ECFP4
def smiles_to_ecfp4(smiles, n_bits=1022):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp)
    else:
        return None  # Return None for invalid SMILES

# Draw structure
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    else:
        return None

st.title("QSAR Model: pIC50 Prediction")

# --- Single SMILES prediction ---
st.header("?? Single Molecule Prediction")
smiles = st.text_input("Enter SMILES string:")

if smiles:
    st.subheader("Structure:")
    mol_img = draw_molecule(smiles)
    if mol_img:
        st.image(mol_img, use_container_width=False)
    else:
        st.error("Invalid SMILES. Cannot generate structure.")

if st.button("Predict"):
    if smiles:
        fp = smiles_to_ecfp4(smiles)
        if fp is not None:
            fp = fp.reshape(1, -1)
            prediction = model.predict(fp)[0]
            st.success(f"Predicted pIC50: {prediction:.3f}")
        else:
            st.error("Invalid SMILES entered.")

# --- Batch Prediction from CSV ---
st.header("?? Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with 'SMILES' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "SMILES" in df.columns:
        predictions = []
        for sm in df["SMILES"]:
            fp = smiles_to_ecfp4(sm)
            if fp is not None:
                pred = model.predict(fp.reshape(1, -1))[0]
            else:
                pred = "Invalid SMILES"
            predictions.append(pred)
        
        df["Predicted_pIC50"] = predictions
        st.success("Batch prediction complete!")
        st.dataframe(df)

        # Download link
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "qsar_predictions.csv", "text/csv")

    else:
        st.error("CSV must contain a 'SMILES' column.")
