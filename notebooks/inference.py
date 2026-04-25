import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH       = "Sepsis Prediction Dataset.csv"
TARGET         = "SepsisLabel"
GROUP          = "Patient_ID"
HOUR_COL       = "Hour"

N_STEPS            = 16
MIN_SEQ_T          = 6
PREDICTION_WINDOW  = 6

TEST_SIZE      = 0.20
VAL_SIZE       = 0.10
RANDOM_STATE   = 42

BATCH_SIZE     = 128
EPOCHS         = 50
PATIENCE       = 8
LEARNING_RATE  = 2e-4

NEGATIVE_MULTIPLIER  = 3
MAX_COL_MISSING      = 0.70
MAX_NEG_PER_PATIENT  = 20


FOCAL_GAMMA    = 2.0
FOCAL_ALPHA    = 0.5


def load_model():
    return tf.keras.models.load_model(
        "sepsis_gru_attention_model.keras",
        custom_objects={"loss_fn":focal_loss()}
    )
    
def focal_loss(gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        fl      = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(fl)
    return loss_fn

# ──────────────────────────────────────────────────────────────────────────────
# Live simulation
# ──────────────────────────────────────────────────────────────────────────────
def simulate_patient_live(
    patient_id, raw_df, preprocessor, model, calibrator,
    feature_cols, n_steps=N_STEPS, prediction_window=PREDICTION_WINDOW,
    threshold=0.5, group_col=GROUP, hour_col=HOUR_COL,
    target_col=TARGET, smooth_window=7, min_display_t=MIN_SEQ_T,
):

    patient_raw = raw_df[raw_df[group_col] == patient_id].copy()
    if patient_raw.empty:
        print(f"Patient {patient_id} not found.")
        return pd.DataFrame()

    patient_raw  = patient_raw.sort_values(hour_col).reset_index(drop=True)
    patient_proc = preprocessor.transform(patient_raw.copy())

    feats       = patient_proc[feature_cols].to_numpy(dtype=np.float32)
    true_labels = patient_raw[target_col].fillna(0).astype(np.int32).to_numpy()
    hours       = patient_raw[hour_col].to_numpy()

    onset_arr  = np.where(true_labels == 1)[0]
    onset_idx  = int(onset_arr[0]) if onset_arr.size > 0 else None
    onset_hour = hours[onset_idx]  if onset_idx is not None else None

    raw_risks, future_labels = [], []
    print(f"\n--- Live Simulation: Patient {patient_id} ---")

    for t in range(len(feats)):
        start = max(0, t - n_steps + 1)
        seq   = feats[start: t + 1]
        if seq.shape[0] < n_steps:
            pad = np.repeat(seq[:1], n_steps - seq.shape[0], axis=0)
            seq = np.vstack([pad, seq])
        raw_risks.append(float(model.predict(seq[None], verbose=0)[0, 0]))
        y_fut = 0
        if onset_idx is not None:
            delta = onset_idx - t
            y_fut = 1 if 1 <= delta <= prediction_window else 0
        future_labels.append(y_fut)

    raw_arr   = np.array(raw_risks, dtype=np.float32)
    cal_risks = (calibrator.predict_proba(raw_arr.reshape(-1, 1))[:, 1]
                 if calibrator is not None else raw_arr.copy())

    a = 2.0 / (smooth_window + 1)
    smoothed    = np.zeros_like(cal_risks)
    smoothed[0] = cal_risks[0]
    for i in range(1, len(cal_risks)):
        smoothed[i] = a * cal_risks[i] + (1 - a) * smoothed[i - 1]

    for t in range(len(feats)):
        if t < min_display_t:
            continue
        print(f"Hour {int(hours[t]):3d} | Raw {raw_risks[t]:.3f} | "
              f"Smoothed {smoothed[t]:.3f} | "
              f"Fut{prediction_window}h {future_labels[t]} | "
              f"{'*** ALERT ***' if smoothed[t] >= threshold else 'OK'}")

    sim_df = pd.DataFrame({
        "Hour": hours, "RawRisk": raw_risks, "CalibratedRisk": cal_risks,
        "SmoothedRisk": smoothed, "FutureWindowLabel": future_labels,
        "CurrentSepsisLabel": true_labels,
    })

    # ── Plot ──────────────────────────────────────────────────────────────────
    disp = sim_df[sim_df.index >= min_display_t].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    risk_min = max(0.0, disp["SmoothedRisk"].min() - 0.05)
    risk_max = min(1.0, disp["SmoothedRisk"].max() + 0.10)

    ax.fill_between(disp["Hour"], risk_min, disp["SmoothedRisk"],
                    alpha=0.20, color="steelblue")
    ax.plot(disp["Hour"], disp["SmoothedRisk"],
            color="steelblue", linewidth=2.5, label="Smoothed Risk (EMA-7)")
    ax.plot(disp["Hour"], disp["CalibratedRisk"],
            linestyle="--", alpha=0.4, linewidth=1, color="steelblue",
            label="Raw Risk (per hour)")
    ax.axhline(threshold, color="darkorange", linestyle="--",
               linewidth=1.8, label=f"Alert Threshold ({threshold:.2f})")

    pos_mask = disp["FutureWindowLabel"].values == 1
    if pos_mask.any():
        ax.scatter(disp["Hour"][pos_mask], disp["SmoothedRisk"][pos_mask],
                   marker="x", s=100, color="darkorange", zorder=5, linewidths=2,
                   label=f"True pre-sepsis window (next {prediction_window}h)")

    if onset_hour is not None:
        ax.axvline(onset_hour, color="red", linestyle=":",
                   linewidth=2, label=f"Sepsis Onset (hour {int(onset_hour)})")

        ax.axvspan(onset_hour - prediction_window, onset_hour,
                   alpha=0.08, color="red", label=f"Pre-onset {prediction_window}h window")

    ax.set_ylim(risk_min, risk_max)
    ax.set_xlim(disp["Hour"].min(), disp["Hour"].max() + 1)
    ax.set_title(f"Live Sepsis Risk — Patient {patient_id}  "
                 f"(onset h={int(onset_hour) if onset_hour is not None else 'none'})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour in ICU", fontsize=12)
    ax.set_ylabel("Sepsis Risk Probability", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"simulation_patient_{patient_id}.png", dpi=150)
    plt.show()

    valid = smoothed[min_display_t:]
    print(f"\nSummary (from h={min_display_t}) | "
          f"mean={valid.mean():.4f}  std={valid.std():.4f}  "
          f"min={valid.min():.4f}  max={valid.max():.4f}")
    if onset_hour is not None:
        first_alert = next(
            (int(hours[t]) for t in range(min_display_t, len(smoothed))
             if smoothed[t] >= threshold), None
        )
        if first_alert is not None:
            lead = int(onset_hour) - first_alert
            print(f"First alert: hour {first_alert}  "
                  f"({'%d hours BEFORE onset' % lead if lead > 0 else 'after onset'})")
        else:
            print("No alert triggered before onset.")
    return sim_df

class CausalPreprocessor:


    ROLL_WINDOWS = [4, 8]

    def __init__(self, target_col=TARGET, group_col=GROUP,
                 hour_col=HOUR_COL, max_col_missing=MAX_COL_MISSING):
        self.target_col      = target_col
        self.group_col       = group_col
        self.hour_col        = hour_col
        self.max_col_missing = max_col_missing
        self.base_cols    = None
        self.feature_cols = None
        self.medians      = None
        self.scaler       = None

    def _sort(self, df):
        return df.sort_values([self.group_col, self.hour_col]).reset_index(drop=True)

    def _build_features(self, df):

        grp        = df.groupby(self.group_col, sort=False)
        new_frames = []

        diffs = grp[self.base_cols].diff().fillna(0.0).astype(np.float32)
        diffs.columns = [f"{c}_diff" for c in self.base_cols]
        new_frames.append(diffs)


        for w in self.ROLL_WINDOWS:
            roll  = grp[self.base_cols].rolling(window=w, min_periods=1)
            rmean = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
            rstd  = roll.std(ddof=0).fillna(0.0).reset_index(level=0, drop=True).astype(np.float32)
            rmean.columns = [f"{c}_rmean{w}" for c in self.base_cols]
            rstd.columns  = [f"{c}_rstd{w}"  for c in self.base_cols]
            new_frames.append(rmean)
            new_frames.append(rstd)

        extra = pd.concat(new_frames, axis=1)
        return pd.concat([df, extra], axis=1)

    def fit(self, df_train):
        df       = self._sort(df_train.copy())
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude  = {self.target_col, self.group_col, self.hour_col}
        candidate = [c for c in num_cols if c not in exclude]

        miss_rate      = df[candidate].isna().mean()
        self.base_cols = [c for c in candidate if miss_rate[c] <= self.max_col_missing]
        if not self.base_cols:
            raise ValueError("No usable feature columns after missing-rate filtering.")

        for c in self.base_cols:
            df[c] = df.groupby(self.group_col, sort=False)[c].ffill()
        self.medians = df[self.base_cols].median(numeric_only=True)
        df[self.base_cols] = df[self.base_cols].fillna(self.medians)

        df = self._build_features(df)

        diff_cols  = [f"{c}_diff"      for c in self.base_cols]
        roll_cols  = ([f"{c}_rmean{w}" for c in self.base_cols for w in self.ROLL_WINDOWS] +
                      [f"{c}_rstd{w}"  for c in self.base_cols for w in self.ROLL_WINDOWS])
        scale_cols = self.base_cols + diff_cols + roll_cols

        self.feature_cols = scale_cols
        self.scaler = StandardScaler()
        self.scaler.fit(df[scale_cols].astype(np.float32))
        return self

    def transform(self, df):
        if self.scaler is None:
            raise RuntimeError("Preprocessor not fitted.")
        out = self._sort(df.copy())
        for c in self.base_cols:
            if c not in out.columns:
                out[c] = np.nan
            out[c] = out.groupby(self.group_col, sort=False)[c].ffill()
        out[self.base_cols] = out[self.base_cols].fillna(self.medians)
        out = self._build_features(out)
        out[self.feature_cols] = self.scaler.transform(
            out[self.feature_cols].astype(np.float32)
        ).astype(np.float32)
        if self.target_col in out.columns:
            out[self.target_col] = out[self.target_col].fillna(0).astype(np.int32)
        return out
    
    
