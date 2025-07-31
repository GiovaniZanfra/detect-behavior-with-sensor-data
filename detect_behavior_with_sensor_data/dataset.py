import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Paths
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
ROT_MISSING_THRESHOLD = 0.2  # sequências com >20% de rot_* missing serão descartadas


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz o uso de memória convertendo colunas numéricas para tipos menores.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df


def add_world_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte aceleração do device frame para world frame e calcula linear acceleration.
    """
    rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    mask = df[rot_cols].notna().all(axis=1)
    if mask.sum() == 0:
        return df  # nada a fazer se não tiver rotações

    quat = df.loc[mask, rot_cols].to_numpy()
    acc = df.loc[mask, acc_cols].to_numpy()
    r = R.from_quat(quat)
    acc_world = r.apply(acc)

    # Atribui aceleração no world frame e aceleração linear
    df.loc[mask, ["accw_x", "accw_y", "accw_z"]] = acc_world
    df.loc[mask, ["linacc_x", "linacc_y", "linacc_z"]] = acc_world - np.array([0, 0, 9.81], dtype=acc_world.dtype)
    return df


def clean_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica pré-processamento em uma única sequência:
    - Filtra por missing em rot_
    - Interpola pequenos gaps em thermopiles e ToF
    - Converte aceleração pro world frame
    - Downcast
    """
    # Descartar sequência se muitos rot_* missing
    rot_missing_frac = df[["rot_w", "rot_x", "rot_y", "rot_z"]].isna().mean().mean()
    if rot_missing_frac > ROT_MISSING_THRESHOLD:
        return None

    # Interpolação em termopiles e ToF (colunas que começam com 'thm_' ou 'tof_')
    sensor_cols = [c for c in df.columns if c.startswith(('thm_', 'tof_'))]
    df[sensor_cols] = df[sensor_cols].interpolate(limit=5).fillna(0)

    # Adiciona world-frame accel
    df = add_world_acc(df)

    # Downcast para economizar memória
    df = downcast_df(df)
    return df


def process_all():
    """
    Lê raw CSVs e processa todas as sequências, salvando em interim/ como Parquet.
    """
    train = pd.read_csv(RAW_DIR / 'train.csv')
    seq_ids = train['sequence_id'].unique()

    for seq in seq_ids:
        seq_df = train[train['sequence_id'] == seq].copy()
        cleaned = clean_sequence(seq_df)
        if cleaned is None:
            continue  # descarta
        out_path = INTERIM_DIR / f'{seq}.parquet'
        cleaned.to_parquet(out_path, index=False)

    # Também copia demographics (apenas downcast e parquet)
    demo = pd.read_csv(RAW_DIR / 'train_demographics.csv')
    demo = downcast_df(demo)
    demo.to_parquet(INTERIM_DIR / 'train_demographics.parquet', index=False)


if __name__ == '__main__':
    process_all()
    print('Processamento raw → interim concluído.')
