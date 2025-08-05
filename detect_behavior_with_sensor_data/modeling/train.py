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
df = pd.read_csv(FEATURES_PATH, index_col=0)
X = df.drop(columns=["gesture", "sequence_type", "subject"])
y_gesture = df["gesture"].values
y_binary = (df["sequence_type"] == "target").astype(int).values
subjects = df["subject"].values

# Label encode gesture for multiclass
gesture_le = LabelEncoder()
y_multiclass = gesture_le.fit_transform(y_gesture)
num_classes = len(gesture_le.classes_)
joblib.dump(gesture_le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# Identify target class indices explicitly
target_gestures = df.loc[df['sequence_type']=='target', 'gesture'].unique()
target_class_indices = gesture_le.transform(target_gestures)

# Cross-validation
cv = GroupKFold(n_splits=N_SPLITS)
scores_binary = []
scores_macro = []
oof_preds = np.zeros((len(X), num_classes))
best_iters = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, groups=subjects)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_mc = y_multiclass[train_idx]
    y_val_mc   = y_multiclass[val_idx]
    y_train_bin= y_binary[train_idx]
    y_val_bin  = y_binary[val_idx]

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=1000,
        random_state=RANDOM_STATE
    )
    model.fit(
        X_train, y_train_mc,
        eval_set=[(X_val, y_val_mc)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50)],
    )

    best_iters.append(model.best_iteration_)

    # OOF predictions
    preds_proba = model.predict_proba(X_val)
    oof_preds[val_idx] = preds_proba
    preds_mc = np.argmax(preds_proba, axis=1)

    # Clear binary conversion using explicit target indices
    preds_bin = np.isin(preds_mc, target_class_indices).astype(int)

    # Metrics
    f1_bin   = f1_score(y_val_bin, preds_bin)
    f1_macro = f1_score(y_val_mc, preds_mc, average='macro')
    scores_binary.append(f1_bin)
    scores_macro.append(f1_macro)

    print(f"Fold {fold}: Binary-F1={f1_bin:.4f}, Macro-F1={f1_macro:.4f}, BestIter={model.best_iteration_}")
    joblib.dump(model, os.path.join(MODEL_DIR, f"lgbm_fold{fold}.pkl"))

print(f"\nCV Binary-F1: {np.mean(scores_binary):.4f} ± {np.std(scores_binary):.4f}")
print(f"CV Macro-F1: {np.mean(scores_macro):.4f} ± {np.std(scores_macro):.4f}")
avg_best_iter = int(np.mean(best_iters))
print(f"Average best iteration: {avg_best_iter}")

# Save OOF preds
oof_df = pd.DataFrame(oof_preds, index=X.index, columns=gesture_le.classes_)
oof_df.to_csv(os.path.join(MODEL_DIR, 'oof_preds.csv'))

# Train final on full data
print("Training final model on full dataset...")
full_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=num_classes,
    learning_rate=0.1,
    num_leaves=31,
    n_estimators=avg_best_iter,
    random_state=RANDOM_STATE
)
full_model.fit(X, y_multiclass)
joblib.dump(full_model, os.path.join(MODEL_DIR, 'lgbm_full.pkl'))
print("Final model trained and saved as lgbm_full.pkl")
