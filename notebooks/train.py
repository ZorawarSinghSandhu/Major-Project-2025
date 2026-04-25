import os
import gc
import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json


from inference import CausalPreprocessor, focal_loss, simulate_patient_live

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
)


import tensorflow as tf
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Conv1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

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


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────



def _extract_patient_sequences(g, feature_cols, hour_col, target_col,
                               n_steps, prediction_window,
                               max_neg_per_patient, dtype, min_t=0):
    g     = g.sort_values(hour_col)
    feats = g[feature_cols].to_numpy(dtype=np.float32)
    labs  = g[target_col].to_numpy(dtype=np.int32)

    onset_arr = np.where(labs == 1)[0]
    onset_idx = int(onset_arr[0]) if onset_arr.size > 0 else None

    if onset_idx is None:
        n = len(g)
        t_indices = (np.linspace(0, n - 1, max_neg_per_patient, dtype=int)
                     if n > max_neg_per_patient else np.arange(n))
    else:
        t_indices = np.arange(min(onset_idx + 1, len(g)))

    seqs = []
    for t in t_indices:
        if t < min_t:
            continue
        start  = max(0, t - n_steps + 1)
        window = feats[start: t + 1]
        if window.shape[0] < n_steps:
            pad    = np.repeat(window[:1], n_steps - window.shape[0], axis=0)
            window = np.vstack([pad, window])
        y = 0
        if onset_idx is not None:
            delta = onset_idx - t
            y = 1 if 1 <= delta <= prediction_window else 0
        seqs.append((window.astype(dtype), np.int32(y)))
    return seqs, onset_idx is not None


def create_sequences(
    df, feature_cols,
    n_steps=N_STEPS, prediction_window=PREDICTION_WINDOW,
    target_col=TARGET, group_col=GROUP, hour_col=HOUR_COL,
    max_neg_per_patient=MAX_NEG_PER_PATIENT,
    max_samples=None,
    min_t=MIN_SEQ_T,
    dtype=np.float16,
    rng_seed=RANDOM_STATE,
):

    pos_seqs, neg_seqs = [], []
    rng = np.random.RandomState(rng_seed)

    for pid, g in df.groupby(group_col, sort=False):
        seqs, is_septic = _extract_patient_sequences(
            g, feature_cols, hour_col, target_col,
            n_steps, prediction_window, max_neg_per_patient, dtype,
            min_t=min_t,
        )
        if is_septic:
            pos_seqs.extend(seqs)
        else:
            neg_seqs.extend(seqs)

    if max_samples is not None and len(neg_seqs) > 0:
        neg_budget = max(0, max_samples - len(pos_seqs))
        if len(neg_seqs) > neg_budget:
            idx      = rng.choice(len(neg_seqs), neg_budget, replace=False)
            neg_seqs = [neg_seqs[i] for i in idx]

    combined = pos_seqs + neg_seqs
    perm     = rng.permutation(len(combined))
    combined = [combined[i] for i in perm]

    if not combined:
        raise RuntimeError("create_sequences produced 0 sequences — check your data split.")

    X = np.asarray([s[0] for s in combined], dtype=dtype)
    y = np.asarray([s[1] for s in combined], dtype=np.int32)

    pos    = int((y == 1).sum())
    neg    = int((y == 0).sum())
    ram_mb = X.nbytes / 1024 / 1024
    print(f"  Sequences: {len(y):,} | Pos: {pos:,} | Neg: {neg:,} | "
          f"Pos ratio: {pos / max(1, len(y)):.4f} | RAM: {ram_mb:.0f} MB")
    if pos == 0:
        raise RuntimeError("No positive sequences — check data splits or PREDICTION_WINDOW.")
    return X, y


def make_eval_dataset(X, y, batch_size=BATCH_SIZE):

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(lambda x, lbl: (tf.cast(x, tf.float32), lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_train_dataset(X, y, batch_size=BATCH_SIZE):

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()
    ds = ds.shuffle(min(50_000, len(X)), seed=RANDOM_STATE,
                    reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(lambda x, lbl: (tf.cast(x, tf.float32), lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    steps_per_epoch = max(1, len(X) // batch_size)
    return ds.prefetch(tf.data.AUTOTUNE), steps_per_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
def build_gru_attention_model(input_shape):
    inp = Input(shape=input_shape, name="input")
    x   = Conv1D(64, kernel_size=3, padding="causal",
                 activation="relu", name="conv1d")(inp)
    x   = LayerNormalization()(x)

    # recurrent_dropout disabled: it forces slow non-cuDNN path on GPU
    gru1     = GRU(128, return_sequences=True, dropout=0.20,
                   name="gru1")(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32,
                                  dropout=0.15, name="mha")(gru1, gru1)
    attn_out = LayerNormalization()(Add()([gru1, attn_out]))

    gru2   = GRU(64, return_sequences=True, dropout=0.20, name="gru2")(attn_out)
    gru2   = LayerNormalization()(gru2)
    pooled = GlobalAveragePooling1D(name="gap")(gru2)

    # L2 regularisation on Dense layers to combat train/val AUROC gap
    reg = tf.keras.regularizers.l2(1e-4)
    x   = Dense(64, activation="relu", kernel_regularizer=reg)(pooled)
    x   = Dropout(0.40)(x)
    x   = Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x   = Dropout(0.30)(x)
    out = Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=focal_loss(),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR",  name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Threshold tuning (F2 — recall-weighted for clinical safety)
# ──────────────────────────────────────────────────────────────────────────────
def tune_threshold_f2(y_true, probs):
    best_th, best = 0.5, -1.0
    for th in np.linspace(0.10, 0.70, 121):
        score = fbeta_score(y_true, (probs >= th).astype(int), beta=2, zero_division=0)
        if score > best:
            best, best_th = score, float(th)
            
    
    return best_th, best


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def summarize_metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "Threshold":             threshold,
        "Accuracy":              accuracy_score(y_true, pred),
        "Precision":             precision_score(y_true, pred, zero_division=0),
        "Recall":                recall_score(y_true, pred, zero_division=0),
        "F1":                    f1_score(y_true, pred, zero_division=0),
        "F2":                    fbeta_score(y_true, pred, beta=2, zero_division=0),
        "AUROC":                 roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "AUPRC":                 average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "PredictedPositiveRate": float(pred.mean()),
        "ProbMean":              float(probs.mean()),
        "ProbStd":               float(probs.std()),
    }
    
    
def main():
    set_seed(RANDOM_STATE)
    tf.keras.backend.clear_session()
    gc.collect()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            mixed_precision.set_global_policy("mixed_float16")
            print(f"GPU(s) detected: {len(gpus)} — mixed_float16 enabled")
        except Exception as e:
            print("GPU config error:", e)
    else:
        print("No GPU. Running on CPU.")

    print("Loading data...")
    df = pd.read_csv(
        CSV_PATH, low_memory=False,
        dtype={GROUP: np.int32, TARGET: np.int8, HOUR_COL: np.int16},
        on_bad_lines='skip',   # skip rows with wrong number of fields
    )
    print(f"  (Malformed rows skipped automatically if any)")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(subset=[GROUP, TARGET, HOUR_COL]).copy()
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype(np.float32)
    df[GROUP]  = df[GROUP].astype(np.int32)
    df[TARGET] = df[TARGET].astype(np.int8)
    df = df.sort_values([GROUP, HOUR_COL]).reset_index(drop=True)
    print(f"Rows: {len(df):,} | Cols: {df.shape[1]} | Patients: {df[GROUP].nunique():,}")
    print(f"Positive rate: {df[TARGET].mean():.4f}")

    # Patient-level balance
    ps          = df.groupby(GROUP)[TARGET].max()
    septic_ids  = ps[ps == 1].index.values
    ns_ids      = ps[ps == 0].index.values
    print(f"\nSeptic: {len(septic_ids)} | Non-septic: {len(ns_ids)}")
    n_keep  = min(len(ns_ids), NEGATIVE_MULTIPLIER * len(septic_ids))
    keep_ns = np.random.RandomState(RANDOM_STATE).choice(ns_ids, n_keep, replace=False)
    df      = df[df[GROUP].isin(np.concatenate([septic_ids, keep_ns]))].reset_index(drop=True)
    print(f"Patients kept: {df[GROUP].nunique():,} | Rows: {len(df):,} | "
          f"Pos rate: {df[TARGET].mean():.4f}")

    # Splits
    gss = GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trv_idx, tst_idx = next(gss.split(df, groups=df[GROUP].values))
    df_trv = df.iloc[trv_idx].reset_index(drop=True)
    df_test_raw = df.iloc[tst_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(1, test_size=VAL_SIZE / (1 - TEST_SIZE),
                             random_state=RANDOM_STATE)
    tr_idx, val_idx = next(gss2.split(df_trv, groups=df_trv[GROUP].values))
    df_tr_raw  = df_trv.iloc[tr_idx].reset_index(drop=True)
    df_val_raw = df_trv.iloc[val_idx].reset_index(drop=True)
    print(f"\nPatients | train={df_tr_raw[GROUP].nunique()} "
          f"val={df_val_raw[GROUP].nunique()} "
          f"test={df_test_raw[GROUP].nunique()}")

    # Preprocessing — silent, fast, no fragmentation warnings
    print("\nFitting preprocessor...")
    prep = CausalPreprocessor()
    prep.fit(df_tr_raw)
    feature_cols = prep.feature_cols
    print(f"Feature count: {len(feature_cols)}")

    print("Transforming splits...")
    df_tr   = prep.transform(df_tr_raw)
    df_val  = prep.transform(df_val_raw)
    df_test = prep.transform(df_test_raw)
    gc.collect()

    # ── Sequences ─────────────────────────────────────────────────────────────
    print(f"\nCreating train sequences (MIN_SEQ_T={MIN_SEQ_T})...")
    X_tr, y_tr = create_sequences(
        df_tr, feature_cols,
        max_samples=None,   # no cap — balanced sampler controls ratio
        min_t=MIN_SEQ_T,
        dtype=np.float16,
    )
    if y_tr.sum() == 0:
        raise RuntimeError("No positive train sequences — lower MIN_SEQ_T or PREDICTION_WINDOW.")
    del df_tr; gc.collect()

    print("Creating val sequences (no min_t filter — preserves real distribution)...")
    X_val, y_val = create_sequences(df_val, feature_cols, min_t=0, dtype=np.float16)
    if y_val.sum() == 0:
        raise RuntimeError("No positive val sequences.")
    del df_val; gc.collect()

    print("Creating test sequences...")
    X_test, y_test = create_sequences(df_test, feature_cols, min_t=0, dtype=np.float16)
    del df_test; gc.collect()


    print("\nBuilding train dataset (real distribution + class_weight)...")
    train_ds, steps_per_epoch = make_train_dataset(X_tr, y_tr)


    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    cw_dict = {0: 1.0, 1: pos_weight}
    print(f"  class_weight: {{0: 1.0, 1: {pos_weight:.1f}}} "
          f"({n_pos:,} pos, {n_neg:,} neg)")
    del X_tr; gc.collect()

    val_ds  = make_eval_dataset(X_val,  y_val)
    test_ds = make_eval_dataset(X_test, y_test)
    del X_val, X_test; gc.collect()

    # ── Build & train model ────────────────────────────────────────────────────
    model = build_gru_attention_model((N_STEPS, len(feature_cols)))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_roc_auc", mode="max", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_roc_auc", mode="max", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]
    del df_tr_raw; gc.collect()

    print(f"\nTraining (max {EPOCHS} epochs, early stop patience={PATIENCE})...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=1,
    )

    print("\nEvaluating...")
    val_raw  = model.predict(val_ds,  verbose=0).ravel()
    test_raw = model.predict(test_ds, verbose=0).ravel()

    best_thresh, best_f2 = tune_threshold_f2(y_val, val_raw)
    with open("threshold.json", "w") as f:
        json.dump({"threshold":best_thresh}, f)
    print(f"Best threshold (F2): {best_thresh:.3f}  F2={best_f2:.4f}")

    metrics = summarize_metrics(y_test, test_raw, threshold=best_thresh)
    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>25}: {v:.4f}" if isinstance(v, float) else f"  {k:>25}: {v}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(test_raw, bins=60, alpha=0.7, label="Test probs")
    axes[0].axvline(best_thresh, color="red", linestyle="--",
                    label=f"Threshold {best_thresh:.2f}")
    axes[0].set_title("Raw Probability Distribution"); axes[0].legend()
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Focal Loss"); axes[1].legend()
    plt.tight_layout(); plt.savefig("probability_distributions.png", dpi=150); plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["roc_auc"],     label="Train")
    axes[0].plot(history.history["val_roc_auc"], label="Val")
    axes[0].set_title("AUROC"); axes[0].legend()
    axes[1].plot(history.history["pr_auc"],     label="Train")
    axes[1].plot(history.history["val_pr_auc"], label="Val")
    axes[1].set_title("AUPRC"); axes[1].legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()


    septic_pats = []
    for pid, g in df_test_raw.groupby(GROUP):
        onset = g[g[TARGET] == 1].sort_values(HOUR_COL)
        if len(onset) > 0:
            oh = int(onset.iloc[0][HOUR_COL])
            hb = int((g[HOUR_COL] < oh).sum())
            total = len(g)
            septic_pats.append((pid, oh, hb, total))

    if not septic_pats:
        print("No septic patients in test split.")
        return

    candidates = [(p, o, h, t) for p, o, h, t in septic_pats if 20 <= o <= 60 and h >= 16]
    if not candidates:
        candidates = [(p, o, h, t) for p, o, h, t in septic_pats if o >= 12 and h >= 8]
    if not candidates:
        candidates = [(p, o, h, t) for p, o, h, t in septic_pats]
    sel_id, sel_onset, sel_hb, sel_total = max(candidates, key=lambda x: x[2])
    print(f"\nSimulating patient {sel_id} (onset h={sel_onset}, "
          f"{sel_hb}h before onset, {sel_total}h total stay)")

    sim_df = simulate_patient_live(
        patient_id=sel_id, raw_df=df, preprocessor=prep,
        model=model, calibrator=None, feature_cols=feature_cols,
        threshold=best_thresh,
    )
    print(f"Simulated patient {sel_id}: onset={sel_onset}h, {sel_hb}h of pre-onset data")
    sim_df.to_csv(f"simulation_patient_{sel_id}.csv", index=False)
    model.save("sepsis_gru_attention_model.keras")
    joblib.dump(prep, "preprocessor.pkl")
    print("Done.")
    
if __name__ == "__main__":
    main()