from inference import simulate_patient_live, load_model
import pandas as pd
import matplotlib.pyplot as plt
import time
import joblib
import json

CSV_PATH = "Sepsis Prediction Dataset.csv"

# Adaptive threshold settings — must match simulate_all.py
ADAPTIVE_DELTA  = 0.05    # fire when risk rises 5pp above patient's own baseline
BASELINE_WINDOW = (8, 28)  # hours 8-28 used to establish baseline

if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="Sepsis Risk Monitor", layout="wide")
    st.title("🧠 Live Sepsis Risk Monitor")
    st.caption(
        "Monitors ICU patient vitals hour-by-hour and raises an alert when "
        "sepsis risk rises significantly above the patient's own baseline."
    )

    @st.cache_resource
    def load_everything():
        model = load_model()
        df    = pd.read_csv(CSV_PATH, on_bad_lines="skip", low_memory=False)
        prep  = joblib.load("preprocessor.pkl")
        with open("threshold.json") as f:
            threshold = json.load(f)["threshold"]
        return model, df, prep, threshold

    model, df, prep, threshold = load_everything()
    feature_cols = prep.feature_cols

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        patient_id = st.number_input("Patient ID", value=5935, step=1)
        playback_speed = st.slider("Playback speed (sec/hour)", 0.1, 1.0, 0.3, 0.05)
        st.markdown("---")
        st.subheader("Alert Settings")
        adaptive_delta = st.slider(
            "Alert delta (Δ above baseline)",
            min_value=0.02, max_value=0.15, value=ADAPTIVE_DELTA, step=0.01,
            help="Fire alert when risk rises this much above the patient's own baseline"
        )
        st.caption(
            f"Baseline computed from hours {BASELINE_WINDOW[0]}-{BASELINE_WINDOW[1]}. "
            f"Alert fires when smoothed risk > baseline + {adaptive_delta:.2f}."
        )

    if st.button("▶  Start Simulation", type="primary"):

        # Run full simulation first (computes all hours)
        sim_df = simulate_patient_live(
            patient_id=patient_id,
            raw_df=df,
            preprocessor=prep,
            model=model,
            calibrator=None,
            feature_cols=feature_cols,
            threshold=threshold,
            adaptive_delta=adaptive_delta,
            baseline_window=BASELINE_WINDOW,
        )

        if sim_df.empty:
            st.error(f"Patient {patient_id} not found in dataset.")
            st.stop()

        # Compute patient baseline and effective threshold for display
        hours_arr  = sim_df["Hour"].values
        smooth_arr = sim_df["SmoothedRisk"].values
        baseline_idx = [i for i, h in enumerate(hours_arr)
                        if BASELINE_WINDOW[0] <= h < BASELINE_WINDOW[1]]
        if len(baseline_idx) >= 3:
            patient_baseline = float(sum(smooth_arr[i] for i in baseline_idx) / len(baseline_idx))
        else:
            patient_baseline = float(smooth_arr[:max(1, len(smooth_arr)//3)].mean())
        eff_threshold = patient_baseline + adaptive_delta

        # Check if septic
        onset_rows = sim_df[sim_df["CurrentSepsisLabel"] == 1]
        onset_hour = int(sim_df.loc[sim_df["CurrentSepsisLabel"] == 1, "Hour"].min()) \
                     if not onset_rows.empty else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Patient ID", patient_id)
        col2.metric("Baseline Risk", f"{patient_baseline:.3f}")
        col3.metric("Alert Threshold", f"{eff_threshold:.3f}",
                    delta=f"+{adaptive_delta:.2f} above baseline")

        # ── Live chart playback ────────────────────────────────────────────────
        chart_placeholder  = st.empty()
        metric_placeholder = st.empty()
        status_placeholder = st.empty()

        hours_seen = []
        risks_seen = []

        fig, ax = plt.subplots(figsize=(12, 5))

        for i in range(len(sim_df)):
            h    = sim_df["Hour"].iloc[i]
            risk = sim_df["SmoothedRisk"].iloc[i]
            raw  = sim_df["RawRisk"].iloc[i]
            hours_seen.append(h)
            risks_seen.append(risk)

            ax.clear()
            risk_min = max(0.0, min(risks_seen) - 0.05)
            risk_max = min(1.0, max(risks_seen) + 0.10)

            ax.fill_between(hours_seen, risk_min, risks_seen, alpha=0.20,
                            color="steelblue")
            ax.plot(hours_seen, sim_df["RawRisk"].iloc[:i+1].tolist(),
                    linestyle="--", alpha=0.35, color="gray", lw=1,
                    label="Raw (per hour)")
            ax.plot(hours_seen, risks_seen, color="steelblue", lw=2.2,
                    label="Smoothed Risk (EMA-7)")
            ax.axhline(eff_threshold, color="darkorange", linestyle="--", lw=1.6,
                       label=f"Alert threshold ({eff_threshold:.2f})")

            if onset_hour is not None:
                ax.axvline(onset_hour, color="red", linestyle=":", lw=1.5,
                           label=f"Sepsis onset (h={onset_hour})")

            ax.set_ylim(risk_min, risk_max)
            ax.set_title(
                f"Patient {patient_id} — "
                f"{'SEPTIC (onset h=' + str(onset_hour) + ')' if onset_hour else 'NON-SEPTIC'}",
                fontsize=13, fontweight="bold"
            )
            ax.set_xlabel("Hour in ICU"); ax.set_ylabel("Sepsis Risk Probability")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            chart_placeholder.pyplot(fig)

            # Metrics
            current_pct  = risk * 100
            prev_risk    = risks_seen[-2] if len(risks_seen) > 1 else risk
            delta_pct    = (risk - prev_risk) * 100
            metric_placeholder.metric(
                label="Current Sepsis Risk",
                value=f"{current_pct:.2f}%",
                delta=f"{delta_pct:+.2f}%"
            )

            # Status badge
            margin = risk - eff_threshold
            if margin >= 0:
                status_placeholder.error(
                    f"🚨 HIGH RISK — alert threshold exceeded "
                    f"(+{margin*100:.1f}pp above threshold)"
                )
            elif margin > -0.03:
                status_placeholder.warning(
                    f"⚠️  ELEVATED RISK — approaching threshold "
                    f"({abs(margin)*100:.1f}pp below)"
                )
            else:
                status_placeholder.success(
                    f"✅  LOW RISK — {abs(margin)*100:.1f}pp below threshold"
                )

            time.sleep(playback_speed)

        plt.close(fig)
        st.success("Simulation complete.")
        
        
        
        
onset_arr = np.where(labs == 1)[0]
onset_idx = int(onset_arr[0]) if onset_arr.size > 0 else None

for t in t_indices:
    start  = max(0, t - n_steps + 1)
    window = feats[start: t + 1]

    if window.shape[0] < n_steps:
        pad = np.repeat(window[:1],
                        n_steps - window.shape[0],
                        axis=0)
        window = np.vstack([pad, window])

    if onset_idx is not None:
        delta = onset_idx - t
        y = 1 if 1 <= delta <= prediction_window else 0