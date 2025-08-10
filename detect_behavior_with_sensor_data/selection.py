import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

# Import project config and selection params
from detect_behavior_with_sensor_data.config import (  # agora definimos constantes no config.py
    LASSO_C,
    MODELS_DIR,
    N_BOOTSTRAPS,
    N_LASSO_FEATS,
    N_SFS_FEATS,
    PROCESSED_DATA_DIR,
    SFS_CV_FOLDS,
)


def stable_lasso_selection(X, y):
    """
    Stability selection via L1 Logistic (lasso) bootstrap.
    Usa constantes de config: N_BOOTSTRAPS, N_LASSO_FEATS, LASSO_C.
    """
    sel_counts = pd.Series(0, index=X.columns)
    rng = np.random.RandomState(42)
    n = X.shape[0]
    for i in range(N_BOOTSTRAPS):
        idx = rng.choice(n, size=int(n * 0.8), replace=False)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        l1 = LogisticRegression(penalty='l1', solver='saga', C=LASSO_C,
                                max_iter=2000, random_state=42 + i)
        # Pula se só houver uma classe
        if yb.nunique() < 2:
            continue
        l1.fit(Xb, yb)
        support = np.abs(l1.coef_).ravel() > 1e-6
        print(support)
        sel_counts[support] += 1
    freqs = (sel_counts / N_BOOTSTRAPS).sort_values(ascending=False)
    return freqs.head(N_LASSO_FEATS).index.to_list()


def sfs_selection(X, y, groups):
    """
    Sequential forward selection based on f1_macro.
    Usa constantes: N_SFS_FEATS, SFS_CV_FOLDS.
    """
    cv = StratifiedGroupKFold(n_splits=SFS_CV_FOLDS)
    from sklearn.ensemble import RandomForestClassifier
    base_est = RandomForestClassifier(n_estimators=200, random_state=42)
    sfs = SequentialFeatureSelector(
        base_est,
        n_features_to_select=N_SFS_FEATS,
        direction='forward',
        scoring='f1_macro',
        cv=cv.split(X, y, groups),
        n_jobs=-1
    )
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])


def run_selection_pipeline():
    # paths da config
    feat_file = PROCESSED_DATA_DIR / 'features_train.csv'
    output_path = MODELS_DIR / 'feats_stable_sfs.pkl'

    df = pd.read_csv(feat_file, index_col=0)
    X = df.drop(columns=['gesture', 'sequence_type', 'subject'])
    y = (df['sequence_type'] == 'target').astype(int)
    groups = df['subject']

    print(f"[*] Lasso stability selection: {N_BOOTSTRAPS} bootstraps, top {N_LASSO_FEATS} feats")
    stable_feats = stable_lasso_selection(X, y)
    print(stable_feats)
    print(f"    -> {len(stable_feats)} features")

    print(f"[*] SFS forward selection: top {N_SFS_FEATS} feats, cv folds={SFS_CV_FOLDS}")
    X_sub = X[stable_feats]
    final_feats = sfs_selection(X_sub, y, groups)
    print(f"    -> {len(final_feats)} final features")

    os.makedirs(output_path.parent, exist_ok=True)
    joblib.dump(final_feats, output_path)
    print(f"[+] Final features salvos em {output_path}")
    return final_feats


if __name__ == '__main__':
    run_selection_pipeline()
