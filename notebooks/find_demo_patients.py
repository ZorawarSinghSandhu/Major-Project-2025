"""
find_demo_patients.py
=====================
Scans the dataset and picks 6 septic patients with well-spread onset times,
enough pre-onset history, and good data completeness — ideal for the
multi-patient demo simulation.

Run this BEFORE running train.py or app.py. It only reads the CSV.

Output:
  • Prints a ranked table of candidate patients
  • Writes demo_patient_ids.json — imported by simulate_all.py
"""

import json
import numpy as np
import pandas as pd

# ── Config — keep in sync with train.py ────────────────────────────────────
CSV_PATH  = "Sepsis Prediction Dataset.csv"
GROUP     = "Patient_ID"
TARGET    = "SepsisLabel"
HOUR_COL  = "Hour"

N_DEMO_PATIENTS = 6       # How many patients to select for the demo
MIN_ONSET_HOUR  = 8      # Onset must happen at least this late (enough warm-up)
MAX_ONSET_HOUR  = 75      # Avoid outliers with very late onset
MIN_PRE_ONSET_H = 8      # Must have at least this many hours BEFORE onset
MAX_MISSING_PCT = 0.60    # Drop patients where >40% of features are NaN
TARGET_ONSETS   = [20, 30, 40, 50, 60, 72]  # Try to pick one near each of these


def load_and_describe(csv_path):
    df = pd.read_csv(
        csv_path, low_memory=False,
        dtype={GROUP: "Int32", TARGET: "Int8", HOUR_COL: "Int16"},
        on_bad_lines="skip",
    )
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(subset=[GROUP, TARGET, HOUR_COL])
    df[GROUP]  = df[GROUP].astype(np.int32)
    df[TARGET] = df[TARGET].astype(np.int8)
    df[HOUR_COL] = df[HOUR_COL].astype(np.int16)
    df = df.sort_values([GROUP, HOUR_COL]).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows | {df[GROUP].nunique():,} patients")
    return df


def score_patient(pid, g, feature_cols):
    """
    Returns a dict of stats for a septic patient, or None if it fails filters.
    """
    onset_rows = g[g[TARGET] == 1].sort_values(HOUR_COL)
    if onset_rows.empty:
        return None

    onset_hour  = int(onset_rows.iloc[0][HOUR_COL])
    total_hours = len(g)
    pre_onset_h = int((g[HOUR_COL] < onset_hour).sum())

    # Hard filters
    if onset_hour < MIN_ONSET_HOUR or onset_hour > MAX_ONSET_HOUR:
        return None
    if pre_onset_h < MIN_PRE_ONSET_H:
        return None

    # Data completeness (features only, excluding metadata cols)
    feat_data = g[feature_cols]
    missing_pct = feat_data.isna().values.mean()
    if missing_pct > MAX_MISSING_PCT:
        return None

    # Variance check — reject flatliners (all same value across time)
    col_vars = feat_data.var(skipna=True)
    nonzero_var_ratio = (col_vars > 0).mean()

    return {
        "Patient_ID":       pid,
        "Onset_Hour":       onset_hour,
        "Pre_Onset_Hours":  pre_onset_h,
        "Total_Hours":      total_hours,
        "Missing_Pct":      round(float(missing_pct), 3),
        "NonzeroVar_Ratio": round(float(nonzero_var_ratio), 3),
    }


def select_diverse_patients(candidates_df, n=N_DEMO_PATIENTS, targets=TARGET_ONSETS):
    """
    Greedily pick one patient closest to each target onset time.
    Falls back to nearest unchosen candidate if a target bucket is empty.
    """
    chosen = []
    remaining = candidates_df.copy()

    for target in targets:
        if remaining.empty:
            break
        idx = (remaining["Onset_Hour"] - target).abs().idxmin()
        chosen.append(remaining.loc[idx])
        remaining = remaining.drop(idx)

    # If we still need more, take highest pre-onset coverage remaining
    while len(chosen) < n and not remaining.empty:
        idx = remaining["Pre_Onset_Hours"].idxmax()
        chosen.append(remaining.loc[idx])
        remaining = remaining.drop(idx)

    return pd.DataFrame(chosen).reset_index(drop=True)


def main():
    df = load_and_describe(CSV_PATH)

    # Identify feature columns (exclude metadata)
    meta_cols = [GROUP, TARGET, HOUR_COL]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Feature columns: {len(feature_cols)}\n")

    # ── Score every septic patient ──────────────────────────────────────────
    print("Scanning septic patients …")
    records = []
    septic_ids = df.groupby(GROUP)[TARGET].max()
    septic_ids = septic_ids[septic_ids == 1].index

    for pid, g in df[df[GROUP].isin(septic_ids)].groupby(GROUP):
        stats = score_patient(pid, g, feature_cols)
        if stats is not None:
            records.append(stats)

    if not records:
        print("No candidates passed filters — relax MIN/MAX_ONSET_HOUR or MIN_PRE_ONSET_H.")
        return

    candidates = pd.DataFrame(records).sort_values("Onset_Hour").reset_index(drop=True)
    print(f"Candidates passing all filters: {len(candidates)}")

    # ── Print distribution of onset hours ──────────────────────────────────
    print("\n── Onset hour distribution (passing candidates) ──")
    bins = [0, 24, 36, 48, 60, 75, 999]
    labels = ["<24h", "24-36h", "36-48h", "48-60h", "60-75h", ">75h"]
    candidates["Onset_Bin"] = pd.cut(
        candidates["Onset_Hour"], bins=bins, labels=labels, right=False
    )
    print(candidates["Onset_Bin"].value_counts().sort_index().to_string())

    # ── Select diverse patients ─────────────────────────────────────────────
    selected = select_diverse_patients(candidates)

    print("\n── Selected demo patients ─────────────────────────────────────────")
    print(selected[[
        "Patient_ID", "Onset_Hour", "Pre_Onset_Hours",
        "Total_Hours", "Missing_Pct", "NonzeroVar_Ratio"
    ]].to_string(index=False))

    # ── Write JSON ──────────────────────────────────────────────────────────
    out = {
        "patient_ids":   selected["Patient_ID"].astype(int).tolist(),
        "onset_hours":   selected["Onset_Hour"].astype(int).tolist(),
        "pre_onset_hrs": selected["Pre_Onset_Hours"].astype(int).tolist(),
    }
    with open("demo_patient_ids.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n✅ Saved demo_patient_ids.json")
    print(f"   Patient IDs: {out['patient_ids']}")
    print(f"   Onset hours: {out['onset_hours']}")


if __name__ == "__main__":
    main()