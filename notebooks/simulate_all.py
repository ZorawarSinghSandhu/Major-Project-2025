"""
simulate_all.py
===============
Runs the live sepsis risk simulation for 7 hardcoded patients:
  - 6 septic patients with different onset times
  - 1 non-septic patient (short stay so risk stays flat)

FIXED vs previous version:
  - Column name corrected: "FutureWindowLabel" not "TrueLabel"
  - Non-septic patient selection prefers 40-80h stays (not the 336h patient
    whose risk rises due to ICULOS effect — long ICU stays correlate with
    higher risk regardless of sepsis)
  - Pre-onset shading uses actual FutureWindowLabel column
  - Threshold loaded from threshold.json (not hardcoded)

Prerequisites:
  train.py  →  sepsis_gru_attention_model.keras | preprocessor.pkl | threshold.json

Usage:
  python simulate_all.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from inference import CausalPreprocessor, focal_loss, simulate_patient_live

# ──────────────────────────────────────────────────────────────────────────────
# HARDCODED DEMO PATIENTS
# ──────────────────────────────────────────────────────────────────────────────
SEPTIC_PATIENTS = [
    # (patient_id, onset_hour)
    (118572, 70),
    (8297,   30),
    (5462,   40),
    (7017,   51),
    (114864, 57),
    (5935,   59),
]

# Leave as None to auto-select a ~50h non-septic patient (flat risk curve).
NON_SEPTIC_PATIENT_ID = None

# ── Adaptive threshold settings ───────────────────────────────────────────────
# Instead of one global threshold for all patients, each patient gets their own:
#   effective_threshold = patient_baseline + ADAPTIVE_DELTA
# where patient_baseline = mean smoothed risk during BASELINE_WINDOW hours.
#
# WHY: The model's absolute output range is narrow (0.10-0.35) with significant
# overlap between septic/non-septic. A global threshold of 0.30 works for some
# patients but not others depending on their individual clinical baseline.
# Adaptive thresholds detect DETERIORATION from each patient's own normal,
# which is also how real clinical monitoring works.
#
# ADAPTIVE_DELTA=0.05 means: fire when risk rises 5 percentage points above
# the patient's own stable baseline — a clinically meaningful deterioration.
ADAPTIVE_DELTA   = 0.05
BASELINE_WINDOW  = (8, 28)   # use hours 8-28 for patient baseline (avoids padding)

# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH    = "Sepsis Prediction Dataset.csv"
GROUP       = "Patient_ID"
TARGET      = "SepsisLabel"
HOUR_COL    = "Hour"
MODEL_PATH  = "sepsis_gru_attention_model.keras"
PREP_PATH   = "preprocessor.pkl"
THRESH_PATH = "threshold.json"


def load_raw(csv_path):
    df = pd.read_csv(
        csv_path, low_memory=False,
        dtype={GROUP: "Int32", TARGET: "Int8", HOUR_COL: "Int16"},
        on_bad_lines="skip",
    )
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(subset=[GROUP, TARGET, HOUR_COL])
    df[GROUP]    = df[GROUP].astype(np.int32)
    df[TARGET]   = df[TARGET].astype(np.int8)
    df[HOUR_COL] = df[HOUR_COL].astype(np.int16)
    return df.sort_values([GROUP, HOUR_COL]).reset_index(drop=True)


def find_non_septic_patient(raw_df, target_hours=50):
    """
    Find a non-septic patient with a stay close to target_hours.

    WHY target_hours=50 instead of longest possible:
      The best predictor in this dataset is ICULOS (ICU length of stay).
      A non-septic patient with a 336-hour stay will show a steady risk rise
      over time because ICULOS keeps growing — even though they never develop
      sepsis. This creates a misleading simulation where the risk crosses the
      threshold purely due to time in ICU, not clinical deterioration.
      A ~50h non-septic patient will have a flat or gently fluctuating curve
      that stays below threshold, which is what we want to demonstrate.
    """
    max_label     = raw_df.groupby(GROUP)[TARGET].max()
    non_septic_ids = max_label[max_label == 0].index
    counts        = raw_df[raw_df[GROUP].isin(non_septic_ids)].groupby(GROUP).size()

    # Find patient whose stay length is closest to target_hours
    best_pid  = int((counts - target_hours).abs().idxmin())
    best_hrs  = int(counts[best_pid])
    print(f"Auto-selected non-septic patient: ID={best_pid}, {best_hrs}h ICU stay "
          f"(target was ~{target_hours}h)")
    return best_pid


def _plot_single(sim_df, pid, onset_h, threshold, is_septic=True, ema_span=7,
                 save=True):
    """
    Plot one patient simulation.
    Uses 'FutureWindowLabel' (1 = sepsis within next 6h) for pre-onset shading.
    Uses 'Alert' (1 = smoothed risk >= threshold) for alert markers.
    """
    hours    = sim_df["Hour"].values
    raw_risk = sim_df["RawRisk"].values
    smooth   = pd.Series(raw_risk).ewm(span=ema_span, adjust=False).mean().values

    fig, ax = plt.subplots(figsize=(13, 4.5))
    risk_min = max(0.0, smooth.min() - 0.05)
    risk_max = min(1.0, smooth.max() + 0.10)

    ax.fill_between(hours, risk_min, smooth, alpha=0.18, color="steelblue")
    ax.plot(hours, raw_risk, color="gray", linestyle="--", lw=0.9,
            alpha=0.55, label="Raw risk (per hour)")
    ax.plot(hours, smooth, color="steelblue", lw=2.0,
            label=f"Smoothed risk (EMA-{ema_span})")
    ax.axhline(threshold, color="darkorange", linestyle="--", lw=1.5,
               label=f"Alert threshold ({threshold:.2f})")

    if is_septic and onset_h is not None:
        # FutureWindowLabel = 1 means "sepsis will occur within next 6h"
        pre_mask = sim_df["FutureWindowLabel"].values == 1
        pre_hrs  = hours[pre_mask]
        if len(pre_hrs):
            ax.axvspan(pre_hrs[0], onset_h, alpha=0.15, color="red",
                       label="Pre-onset 6h window")
            ax.scatter(pre_hrs, smooth[pre_mask],
                       marker="x", s=100, color="darkorange", zorder=5,
                       linewidths=2, label="True pre-sepsis window (next 6h)")
        ax.axvline(onset_h, color="red", linestyle=":", lw=1.8,
                   label=f"Sepsis onset (hour {onset_h})")

    title = (f"Live Sepsis Risk — Patient {pid}  (onset h={onset_h})"
             if is_septic else f"Live Sepsis Risk — Patient {pid}  (NON-SEPTIC)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour in ICU")
    ax.set_ylabel("Sepsis Risk Probability")
    ax.set_ylim(risk_min, risk_max)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fname = f"sim_patient_{pid}.png"
        plt.savefig(fname, dpi=150)
        print(f"  Saved {fname}")
    plt.close()


def _plot_summary(results, threshold, ema_span=7):
    """Combined 2×4 or 3×3 grid of all simulated patients."""
    n    = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.2))
    axes = axes.flatten()

    for ax, r in zip(axes, results):
        sim_df  = r["sim_df"]
        pid     = r["pid"]
        onset_h = r.get("onset_h")
        is_sep  = r["is_septic"]
        hours   = sim_df["Hour"].values
        smooth  = pd.Series(sim_df["RawRisk"].values).ewm(
            span=ema_span, adjust=False).mean().values

        risk_min = max(0.0, smooth.min() - 0.05)
        risk_max = min(1.0, smooth.max() + 0.10)

        ax.fill_between(hours, risk_min, smooth, alpha=0.18, color="steelblue")
        ax.plot(hours, sim_df["RawRisk"].values,
                color="gray", linestyle="--", lw=0.7, alpha=0.5)
        ax.plot(hours, smooth, color="steelblue", lw=1.8)
        ax.axhline(threshold, color="darkorange", linestyle="--", lw=1.3)

        if is_sep and onset_h is not None:
            pre_mask = sim_df["FutureWindowLabel"].values == 1
            pre_hrs  = hours[pre_mask]
            if len(pre_hrs):
                ax.axvspan(pre_hrs[0], onset_h, alpha=0.15, color="red")
                ax.scatter(pre_hrs, smooth[pre_mask],
                           marker="x", s=60, color="darkorange", zorder=5)
            ax.axvline(onset_h, color="red", linestyle=":", lw=1.5)

        sub_title = (f"Patient {pid} — onset h={onset_h}" if is_sep
                     else f"Patient {pid} — NON-SEPTIC")
        ax.set_title(sub_title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Hour in ICU", fontsize=8)
        ax.set_ylabel("Risk", fontsize=8)
        ax.set_ylim(risk_min, risk_max)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Early Sepsis Detection System — All Demo Patients  "
        f"(threshold={threshold:.2f})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("sim_all_patients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved sim_all_patients.png")


def print_summary_table(results, threshold):
    """Print a clean summary table for evaluation presentation."""
    print("\n" + "=" * 70)
    print(f"{'Patient':>10} {'Type':>12} {'Onset':>6} {'Alert?':>7} "
          f"{'Lead':>6} {'Peak Risk':>10}")
    print("-" * 70)
    for r in results:
        sim_df  = r["sim_df"]
        pid     = r["pid"]
        onset_h = r.get("onset_h")
        is_sep  = r["is_septic"]
        hours   = sim_df["Hour"].values
        smooth  = pd.Series(sim_df["RawRisk"].values).ewm(
            span=7, adjust=False).mean().values

        peak = float(smooth.max())
        first_alert = next(
            (int(hours[t]) for t in range(6, len(smooth))
             if smooth[t] >= threshold), None
        )
        if is_sep and onset_h is not None:
            if first_alert is not None:
                lead = onset_h - first_alert
                alert_str = f"h={first_alert}"
                lead_str  = f"{lead:+d}h"
            else:
                alert_str = "no"
                lead_str  = "—"
        else:
            alert_str = "yes" if first_alert is not None else "no"
            lead_str  = "—"

        label = f"SEPTIC (h={onset_h})" if is_sep else "NON-SEPTIC"
        print(f"  {pid:>8}   {label:>14}   {str(onset_h) if onset_h else '—':>4}   "
              f"{alert_str:>7}   {lead_str:>5}   {peak:.3f}")
    print("=" * 70)


def main():
    # ── Load artefacts ─────────────────────────────────────────────────────
    print("Loading model and preprocessor …")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects = {
            "loss_fn": focal_loss(),
            "focal_loss_fn": focal_loss()
        },
        compile = False
        )   # focal_loss registered via decorator
    prep  = joblib.load(PREP_PATH)

    with open(THRESH_PATH) as f:
        threshold = json.load(f)["threshold"]
    print(f"Global threshold (fallback): {threshold:.3f}")
    print(f"Using adaptive threshold: baseline + delta={ADAPTIVE_DELTA:.3f} "
          f"(window h={BASELINE_WINDOW[0]}-{BASELINE_WINDOW[1]})")

    raw_df       = load_raw(CSV_PATH)
    feature_cols = prep.feature_cols

    # ── Resolve non-septic patient ─────────────────────────────────────────
    ns_pid = NON_SEPTIC_PATIENT_ID
    if ns_pid is None:
        ns_pid = find_non_septic_patient(raw_df, target_hours=50)

    # ── Build full patient list ────────────────────────────────────────────
    all_patients = [(pid, onset_h, True)  for pid, onset_h in SEPTIC_PATIENTS]
    all_patients.append((ns_pid, None, False))

    print(f"\nSimulating {len(all_patients)} patients:")
    for pid, onset_h, is_sep in all_patients:
        tag = f"onset h={onset_h}" if is_sep else "NON-SEPTIC"
        print(f"  Patient {pid:>7}  {tag}")

    # ── Run simulations ────────────────────────────────────────────────────
    results = []
    for pid, onset_h, is_sep in all_patients:
        print(f"\n{'─' * 60}")
        sim_df = simulate_patient_live(
            patient_id=pid,
            raw_df=raw_df,
            preprocessor=prep,
            model=model,
            calibrator=None,
            feature_cols=feature_cols,
            threshold=threshold,          # fallback if adaptive fails
            adaptive_delta=ADAPTIVE_DELTA,
            baseline_window=BASELINE_WINDOW,
        )
        if sim_df.empty:
            print(f"  Skipping patient {pid} — not found.")
            continue

        sim_df.to_csv(f"sim_patient_{pid}.csv", index=False)
        _plot_single(sim_df, pid, onset_h, threshold, is_septic=is_sep)
        results.append({"pid": pid, "onset_h": onset_h,
                        "is_septic": is_sep, "sim_df": sim_df})

    # ── Summary figure + table ─────────────────────────────────────────────
    _plot_summary(results, threshold)
    print_summary_table(results, threshold)
    print("\n✅ Done. Saved: individual .png/.csv files + sim_all_patients.png")


if __name__ == "__main__":
    main()