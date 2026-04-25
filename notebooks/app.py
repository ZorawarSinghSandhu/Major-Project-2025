from inference import simulate_patient_live, load_model
import pandas as pd
import matplotlib.pyplot as plt
import time
import joblib
import json

CSV_PATH = "Sepsis Prediction Dataset.csv"
# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────






# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────



if __name__ == "__main__":
    import streamlit as st

    st.title("🧠 Live Sepsis Risk Monitor")

    # LOAD saved model
    @st.cache_resource
    def load_everything():
        model=load_model()
        df=pd.read_csv("Sepsis Prediction Dataset.csv")
        prep = joblib.load("preprocessor.pkl")
        
        with open("threshold.json") as f:
            threshold = json.load(f)["threshold"]
        
        return model, df, prep, threshold
    
    model, df, prep, threshold = load_everything()
    feature_cols = prep.feature_cols


    patient_id = st.number_input("Patient ID", value=5935)

    

    if st.button("Start Simulation"):

        sim_df = simulate_patient_live(
            patient_id=patient_id,
            raw_df=df,
            preprocessor=prep,
            model=model,
            calibrator=None,
            feature_cols=feature_cols,
            threshold=threshold
        )

        chart_placeholder = st.empty()

        hours = []
        risks = []
        fig, ax = plt.subplots(figsize=(10, 5))

        for i in range(len(sim_df)):
            hours.append(sim_df["Hour"].iloc[i])
            risks.append(sim_df["SmoothedRisk"].iloc[i])
            raw = sim_df["RawRisk"].iloc[:i+1]

            ax.clear()

            # Smooth line (main)
            ax.plot(hours, risks, linewidth=2.5)

            # Raw line (faded)
            ax.plot(hours, raw, linestyle='--', alpha=0.3)

            # Fill area (this makes it look professional)
            ax.fill_between(hours, min(risks), risks, alpha=0.2)

            # Threshold
            ax.axhline(threshold, linestyle='--')

            ax.set_title("Live Sepsis Risk")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Risk")

            chart_placeholder.pyplot(fig)
            time.sleep(0.3)