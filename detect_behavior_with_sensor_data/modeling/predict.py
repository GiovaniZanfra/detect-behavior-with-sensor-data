# modeling/predict.py
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from typing import Any
from loguru import logger

# IMPORT THESE from your package (ensure they are available in the notebook/session)
# from detect_behavior_with_sensor_data.dataset import clean_sequence
# from detect_behavior_with_sensor_data.features import extract_sequence_features
import detect_behavior_with_sensor_data.config as config
from detect_behavior_with_sensor_data.dataset import clean_sequence
from detect_behavior_with_sensor_data.features import extract_sequence_features

# Model files (you can override with env var if wanted)
MODELS_DIR = config.MODELS_DIR

# Try to load model artifacts at import time (gateway will import module once)
def _safe_joblib_load(p: Path):
    try:
        return joblib.load(p)
    except Exception as e:
        logger.warning(f"Could not load {p}: {e}")
        return None

# Primary models
MODEL_FULL_PATH = MODELS_DIR / "lgbm_full.pkl"
MODEL_IMU_PATH  = MODELS_DIR / "lgbm_full_imu_only.pkl"
LE_PATH         = MODELS_DIR / "label_encoder.pkl"
FEATURE_LIST_P  = MODELS_DIR / "features_used_full.json"  # optional: saved feature ordering

model_full = _safe_joblib_load(MODEL_FULL_PATH)
model_imu  = _safe_joblib_load(MODEL_IMU_PATH)
label_enc  = _safe_joblib_load(LE_PATH)

# If features file exists, load it (used to reindex features exactly)
if FEATURE_LIST_P.exists():
    try:
        with open(FEATURE_LIST_P, "r") as fh:
            MODEL_FEATURES = json.load(fh)
    except Exception:
        MODEL_FEATURES = None
else:
    MODEL_FEATURES = None

# Helper conversions
def to_pandas_maybe(obj: Any) -> pd.DataFrame:
    """Accept polars or pandas or list/dict."""
    if obj is None:
        return pd.DataFrame()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    # Series -> DataFrame row
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    # list/dict
    return pd.DataFrame(obj)

def has_nonimu_sensors(df: pd.DataFrame) -> bool:
    thm_cols = [c for c in df.columns if c.startswith("thm_")]
    tof_cols = [c for c in df.columns if c.startswith("tof_")]
    has_thm = any(df[c].notna().any() for c in thm_cols) if thm_cols else False
    has_tof = any((df[c].fillna(-1) >= 0).any() for c in tof_cols) if tof_cols else False
    return has_thm or has_tof

def prepare_features_for_model(feats: pd.DataFrame, model, model_feature_list=None) -> pd.DataFrame:
    """Reindex single-row feats to model features and convert dtypes."""
    # Determine feature order
    model_feats = None
    if model_feature_list:
        model_feats = model_feature_list
    else:
        model_feats = getattr(model, "feature_name_", None)
        if model_feats is None:
            try:
                model_feats = model.booster_.feature_name()
            except Exception:
                model_feats = None
    if model_feats is None:
        model_feats = feats.columns.tolist()
    # Reindex and fill
    X = feats.reindex(columns=model_feats, fill_value=0)
    # ensure numeric dtypes
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X.astype(float)

# Safe fallback label (must be one of train labels)
def safe_label() -> str:
    if label_enc is not None and hasattr(label_enc, "classes_"):
        return label_enc.classes_[0]
    return "Text on phone"  # conservative fallback present in train labels

# Main predict function expected by the Kaggle inference server
def predict(sequence, demographics) -> str:
    """
    sequence: polars.DataFrame or pandas.DataFrame (one sequence of raw timesteps)
    demographics: polars.DataFrame/pandas.DataFrame/Series with 1 row
    returns: gesture string (must be in training labels)
    """
    try:
        seq_df = to_pandas_maybe(sequence)
        demo_df = to_pandas_maybe(demographics)

        # demographics may be DataFrame with 1 row or a Series
        if isinstance(demo_df, pd.DataFrame) and demo_df.shape[0] >= 1:
            demo_series = demo_df.iloc[0]
        elif isinstance(demo_df, pd.Series):
            demo_series = demo_df
        else:
            demo_series = pd.Series(dtype="object")

        if seq_df is None or seq_df.shape[0] == 0:
            return safe_label()

        # Decide model
        use_imu = not has_nonimu_sensors(seq_df)
        model_to_use = model_imu if use_imu and model_imu is not None else model_full
        if model_to_use is None:
            # no model loaded -> fallback
            logger.error("No model available for inference; returning safe label")
            return safe_label()

        # Preprocess and feature extraction (functions must be importable)
        seq_clean = clean_sequence(seq_df)
        if seq_clean is None:
            return safe_label()

        feats = extract_sequence_features(seq_clean).to_frame().T  # single-row DataFrame

        # Merge demographics (if any)
        for c in demo_series.index:
            # Some demographics might be numpy types; keep them as-is
            feats[c] = demo_series[c]

        # Remove metadata columns
        feats = feats.drop(columns=["gesture", "sequence_type", "subject"], errors="ignore")

        # Reindex to model features; prefer saved feature list if available
        use_feature_list = None
        if MODEL_FEATURES:
            # if you saved per-model feature lists you can choose based on model_to_use name
            use_feature_list = MODEL_FEATURES
        X = prepare_features_for_model(feats, model_to_use, model_feature_list=use_feature_list)

        proba = model_to_use.predict_proba(X)
        idx = int(np.argmax(proba, axis=1)[0])
        return label_enc.inverse_transform([idx])[0]

    except Exception as e:
        # Don't propagate - gateway will error out.
        logger.exception("Error during predict() — returning safe label")
        return safe_label()

# If run as script you can do a small local smoke test
if __name__ == "__main__":
    # quick smoke test: load a sample file from RAW and run predict for first sequence
    raw_test = config.RAW_DATA_DIR / "test.csv"
    if raw_test.exists():
        df = pd.read_csv(raw_test)
        seq_id = df['sequence_id'].unique()[0]
        seq = df[df['sequence_id'] == seq_id]
        demo_df = pd.read_csv(config.RAW_DATA_DIR / "test_demographics.csv")
        demo = demo_df[demo_df['subject'] == seq['subject'].iloc[0]]
        print("Sample predict:", predict(seq, demo))
