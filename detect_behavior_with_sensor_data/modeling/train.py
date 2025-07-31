import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# Paths
FEATURES_PATH = os.path.join("data/processed", "features_train.csv")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Config
N_SPLITS = 5
RANDOM_STATE = 42

# Load features
feat_df = pd.read_csv(FEATURES_PATH, index_col=0)
# Prepare data
X = feat_df.drop(columns=["gesture", "sequence_type", "subject"])
y_gesture = feat_df["gesture"].values
y_binary = (feat_df["sequence_type"] == "target").astype(int).values
subjects = feat_df["subject"].values

# Label encode gesture for multiclass
le = LabelEncoder()
y_multiclass = le.fit_transform(y_gesture)
num_classes = len(le.classes_)
# Save label encoder
txt_le = os.path.join(MODEL_DIR, 'label_encoder.pkl')
joblib.dump(le, txt_le)

# Cross-validation
cv = GroupKFold(n_splits=N_SPLITS)
scores_binary = []
scores_macro = []
oof_preds = np.zeros((len(X), num_classes))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, groups=subjects)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_mc, y_val_mc = y_multiclass[train_idx], y_multiclass[val_idx]
    y_train_bin, y_val_bin = y_binary[train_idx], y_binary[val_idx]

    # Define model
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=1000,
        random_state=RANDOM_STATE
    )
    # Fit with early stopping via callbacks
    model.fit(
        X_train, y_train_mc,
        eval_set=[(X_val, y_val_mc)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50)],
        # verbose=False
    )

    # Predict probabilities and classes
    preds_proba = model.predict_proba(X_val)
    oof_preds[val_idx, :] = preds_proba
    preds_mc = np.argmax(preds_proba, axis=1)

    # Convert multiclass to binary
    if 'non_target' in le.classes_:
        non_target_idx = list(le.classes_).index('non_target')
        preds_bin = (preds_mc != non_target_idx).astype(int)
    else:
        preds_bin = (preds_mc >= num_classes // 2).astype(int)

    # Compute metrics
    f1_bin = f1_score(y_val_bin, preds_bin)
    f1_macro = f1_score(y_val_mc, preds_mc, average='macro')
    scores_binary.append(f1_bin)
    scores_macro.append(f1_macro)

    print(f"Fold {fold}: Binary-F1={f1_bin:.4f}, Macro-F1={f1_macro:.4f}")

    # Save fold model
    joblib.dump(model, os.path.join(MODEL_DIR, f"lgbm_fold{fold}.pkl"))

# Final CV results
print("\nCV Binary-F1: %.4f ± %.4f" % (np.mean(scores_binary), np.std(scores_binary)))
print("CV Macro-F1: %.4f ± %.4f" % (np.mean(scores_macro), np.std(scores_macro)))

# Save OOF predictions
oof_df = pd.DataFrame(oof_preds, index=X.index, columns=le.classes_)
oof_df.to_csv(os.path.join(MODEL_DIR, 'oof_preds.csv'))
