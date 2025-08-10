import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# importa config
import detect_behavior_with_sensor_data.config as config

# Paths (usa config)
FEATURES_PATH = config.PROCESSED_DATA_DIR / "features_train.csv"
MODEL_DIR = config.MODELS_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Config
N_SPLITS = config.N_SPLITS
RANDOM_STATE = config.RANDOM_STATE
MODEL_TYPE = config.MODEL_TYPE
MODEL_PARAMS = config.MODEL_PARAMS[MODEL_TYPE].copy()
EARLY_STOPPING = getattr(config, "EARLY_STOPPING_ROUNDS", 50)

def create_model(model_type: str, num_classes: int, **kwargs):
    """Factory function to create different model types"""
    if model_type == "xgboost":
        params = MODEL_PARAMS.copy()
        params.update(kwargs)
        params["num_class"] = num_classes
        return xgb.XGBClassifier(**params, use_label_encoder=False)
    elif model_type == "lightgbm":
        params = MODEL_PARAMS.copy()
        params.update(kwargs)
        params["num_class"] = num_classes
        return lgb.LGBMClassifier(**params)
    elif model_type == "random_forest":
        params = MODEL_PARAMS.copy()
        params.update(kwargs)
        return RandomForestClassifier(**params)
    elif model_type == "logistic_regression":
        params = MODEL_PARAMS.copy()
        params.update(kwargs)
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def fit_model(model, X_train, y_train, X_val=None, y_val=None, model_type=None):
    """Fit model with appropriate method based on model type"""
    if model_type in ["xgboost", "lightgbm"] and X_val is not None and y_val is not None:
        # Gradient boosting models with early stopping
        try:
            if model_type == "xgboost":
                # Try different XGBoost API versions
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=EARLY_STOPPING,
                        verbose=False,
                    )
                except TypeError:
                    # Newer XGBoost versions use callbacks
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING)],
                        verbose=False,
                    )
            else:  # lightgbm
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)]
                )
        except Exception as e:
            # Final fallback - fit without early stopping
            print(f"Warning: Early stopping failed for {model_type}, fitting without it: {e}")
            model.fit(X_train, y_train)
    else:
        # Standard sklearn fit for other models
        model.fit(X_train, y_train)
    
    return model

def get_best_iteration(model, model_type):
    """Get best iteration for gradient boosting models"""
    if model_type == "xgboost":
        return getattr(model, "best_iteration", None) or getattr(model, "best_ntree_limit", None)
    elif model_type == "lightgbm":
        return getattr(model, "best_iteration_", None)
    return None

# Carrega dados
df = pd.read_csv(FEATURES_PATH, index_col=0)
print("Loaded features:", df.shape)

# função util para escolher colunas baseado em config.TRAIN_ON
def select_feature_columns(df: pd.DataFrame, train_on: dict):
    cols = []
    # IMU group
    if train_on.get("imu", False):
        imu_prefixes = ("acc_", "rot_", "linacc_", "acc_mag")
        cols += [c for c in df.columns if c.startswith(imu_prefixes)]
    # Thermopiles
    if train_on.get("thm", False):
        cols += [c for c in df.columns if c.startswith("thm_")]
    # ToF
    if train_on.get("tof", False):
        cols += [c for c in df.columns if c.startswith("tof")]
    # seq length
    if train_on.get("seq_length", False) and "seq_length" in df.columns:
        cols += ["seq_length"]
    # demographics
    if train_on.get("demo", False):
        demo_cols = [
            "adult_child",
            "age",
            "sex",
            "handedness",
            "height_cm",
            "shoulder_to_wrist_cm",
            "elbow_to_wrist_cm",
        ]
        cols += [c for c in demo_cols if c in df.columns]
    # remove duplicates & meta
    cols = list(set(cols))  # remove duplicates
    cols = [c for c in cols if c not in ("gesture", "sequence_type", "subject")]
    return cols

selected_cols = select_feature_columns(df, config.TRAIN_ON)
if len(selected_cols) == 0:
    raise RuntimeError("Nenhuma coluna selecionada para treino — verifique config.TRAIN_ON")

print(f"Selected {len(selected_cols)} feature cols for training")

# Prepare matrices
X = df[selected_cols].fillna(0)
y_gesture = df["gesture"].values
y_binary = (df["sequence_type"] == "target").astype(int).values
subjects = df["subject"].values

# Label encode gesture
gesture_le = LabelEncoder()
y_multiclass = gesture_le.fit_transform(y_gesture)
num_classes = len(gesture_le.classes_)
joblib.dump(gesture_le, MODEL_DIR / "label_encoder.pkl")
print("num classes:", num_classes)

# identify target class indices
target_gestures = df.loc[df['sequence_type']=='target', 'gesture'].unique()
target_class_indices = gesture_le.transform(target_gestures)

# CV
cv = GroupKFold(n_splits=N_SPLITS)
oof_preds = np.zeros((len(X), num_classes))
best_rounds = []
scores_binary = []
scores_macro = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, groups=subjects)):
    print(f"Fold {fold} training...")
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr_mc, y_va_mc = y_multiclass[train_idx], y_multiclass[val_idx]
    y_tr_bin, y_va_bin = y_binary[train_idx], y_binary[val_idx]

    # XGBoost classifier wrapper
    clf = create_model(MODEL_TYPE, num_classes)
    # Fit with early stopping
    clf = fit_model(clf, X_tr, y_tr_mc, X_va, y_va_mc, MODEL_TYPE)

    best_it = get_best_iteration(clf, MODEL_TYPE)
    best_rounds.append(int(best_it) if best_it is not None else MODEL_PARAMS.get("n_estimators", 100))

    proba_va = clf.predict_proba(X_va)
    oof_preds[val_idx] = proba_va
    preds_mc = np.argmax(proba_va, axis=1)
    preds_bin = np.isin(preds_mc, target_class_indices).astype(int)

    f1_bin = f1_score(y_va_bin, preds_bin)
    f1_mac = f1_score(y_va_mc, preds_mc, average='macro')
    scores_binary.append(f1_bin)
    scores_macro.append(f1_mac)

    print(f"Fold {fold}: Binary-F1={f1_bin:.4f}, Macro-F1={f1_mac:.4f}, best_it={best_it}")
    joblib.dump(clf, MODEL_DIR / f"{MODEL_TYPE}_fold{fold}.pkl")

print(f"\nCV Binary-F1: {np.mean(scores_binary):.4f} ± {np.std(scores_binary):.4f}")
print(f"CV Macro-F1: {np.mean(scores_macro):.4f} ± {np.std(scores_macro):.4f}")
avg_best_round = int(np.mean([r for r in best_rounds if r is not None]))
print("Average best round:", avg_best_round)

# Save OOF preds + true + pred
oof_df = pd.DataFrame(oof_preds, index=X.index, columns=gesture_le.classes_)
oof_df['y_true_idx'] = y_multiclass
oof_df['y_pred_idx'] = np.argmax(oof_preds, axis=1)
oof_df['gesture_true'] = y_gesture
oof_df['gesture_pred'] = gesture_le.inverse_transform(oof_df['y_pred_idx'].astype(int))
oof_df.to_csv(MODEL_DIR / "oof_preds.csv")
print("Saved OOF to", MODEL_DIR / "oof_preds.csv")

# Train final model on full dataset
print("Training final model on full dataset...")

final_n_estimators = avg_best_round if avg_best_round > 0 else MODEL_PARAMS.get("n_estimators", 100)
final_params = MODEL_PARAMS.copy()
final_params["n_estimators"] = final_n_estimators

final_clf = create_model(MODEL_TYPE, num_classes, **final_params)
final_clf = fit_model(final_clf, X, y_multiclass, model_type=MODEL_TYPE)

# Mode name for files: 'imu_only' if only imu selected, else 'full'
is_imu_only = config.TRAIN_ON.get("imu", False) and not config.TRAIN_ON.get("thm", False) and not config.TRAIN_ON.get("tof", False)
mode = "imu_only" if is_imu_only else "full"

final_model_path = MODEL_DIR / f"{MODEL_TYPE}_full_{mode}.pkl"
joblib.dump(final_clf, final_model_path)
print("Saved final model to", final_model_path)

# Save list of features used + metadata
if getattr(config, "SAVE_FEATURE_LIST", True):
    feat_file = MODEL_DIR / f"features_used_{mode}.json"
    meta = {
        "features": selected_cols,
        "train_on": config.TRAIN_ON,
        "model_type": MODEL_TYPE,
        "is_imu_only": bool(is_imu_only),
        "num_classes": int(num_classes),
    }
    with open(feat_file, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved feature list & metadata to", feat_file)

# Save a separate model metadata file
meta_file = MODEL_DIR / f"model_metadata_{mode}.json"
model_meta = {
    "model_path": str(final_model_path),
    "model_type": MODEL_TYPE,
    "params": final_params,
    "is_imu_only": bool(is_imu_only),
    "features_file": str(feat_file) if getattr(config, "SAVE_FEATURE_LIST", True) else None,
}
with open(meta_file, "w") as fh:
    json.dump(model_meta, fh, indent=2)
print("Saved model metadata to", meta_file)
