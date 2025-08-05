import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

# Paths
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
FFT_TOP_N = 5
SAMPLE_RATE = 100.0  # Hz
HIST_BINS = 10


def summary_stats(arr):
    return {
        'mean': np.nanmean(arr),
        'std': np.nanstd(arr),
        'min': np.nanmin(arr),
        'max': np.nanmax(arr),
        'skew': skew(arr, nan_policy='omit'),
        'kurtosis': kurtosis(arr, nan_policy='omit')
    }


def fft_top_features(arr, prefix):
    yf = np.abs(rfft(np.nan_to_num(arr)))
    freqs = rfftfreq(len(arr), d=1/SAMPLE_RATE)
    idx = np.argsort(yf)[-FFT_TOP_N:]
    feats = {}
    for rank, i in enumerate(sorted(idx)):
        feats[f'{prefix}_fft_freq{rank}'] = freqs[i]
        feats[f'{prefix}_fft_amp{rank}'] = yf[i]
    return feats


def extract_sequence_features(df):
    feats = {}
    # 1. IMU: acc and rot
    imu_axes = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']
    if all(c in df.columns for c in ['acc_x','acc_y','acc_z']):
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        imu_axes.append('acc_mag')

    for c in imu_axes:
        arr = df[c].values
        for k, v in summary_stats(arr).items(): feats[f'{c}_{k}'] = v
        feats.update(fft_top_features(arr, f'{c}'))

    # 2. World-frame accel: linacc
    if all(c in df.columns for c in ['linacc_x','linacc_y','linacc_z']):
        df['linacc_mag'] = np.sqrt(df['linacc_x']**2 + df['linacc_y']**2 + df['linacc_z']**2)
        lin_axes = ['linacc_x','linacc_y','linacc_z','linacc_mag']
        dt = 1.0 / SAMPLE_RATE
        linvel = np.cumsum(df[['linacc_x','linacc_y','linacc_z']].fillna(0), axis=0) * dt
        feats['linacc_disp_total'] = np.nansum(np.linalg.norm(linvel, axis=1))
        feats['linacc_energy'] = np.nansum(0.5 * np.linalg.norm(linvel, axis=1)**2)
        for c in lin_axes:
            arr = df[c].values
            for k, v in summary_stats(arr).items(): feats[f'{c}_{k}'] = v
            feats.update(fft_top_features(arr, f'{c}'))

    # 3. Thermopiles
    for c in [col for col in df.columns if col.startswith('thm_')]:
        arr = df[c].values
        for k, v in summary_stats(arr).items(): feats[f'{c}_{k}'] = v
        diff = np.diff(arr)
        feats[f'{c}_diff_mean'] = np.nanmean(diff)
        feats[f'{c}_diff_std'] = np.nanstd(diff)
        feats[f'{c}_time_to_max'] = np.argmax(arr)

    # 4. Time-of-Flight sensors
    for i in range(1,6):
        tof_cols = [f'tof_{i}_v{j}' for j in range(64) if f'tof_{i}_v{j}' in df.columns]
        if not tof_cols: continue
        flat = df[tof_cols].fillna(-1).values.flatten()
        for k, v in summary_stats(flat).items(): feats[f'tof{i}_{k}'] = v
        flat_pos = flat[flat >= 0]
        hist, _ = np.histogram(flat_pos, bins=HIST_BINS, range=(0,254))
        hist = hist / (hist.sum()+1e-6)
        for b, val in enumerate(hist): feats[f'tof{i}_hist_{b}'] = val
        feats[f'tof{i}_texture'] = np.nanmean(np.abs(flat[1:] - flat[:-1]))

    # 5. Phase dynamics
    total = len(df)
    for ph, cnt in df['behavior'].value_counts().items():
        key = ph.replace(' ', '_')
        feats[f'phase_{key}_cnt'] = cnt
        feats[f'phase_{key}_frac'] = cnt / total

    # 6. Orientation one-hot
    ori = df['orientation'].iloc[0]
    # Clean name
    ori_clean = ori.lower().replace(' ', '_').replace('-', '').replace('/', '_')
    feats[f'ori_{ori_clean}'] = 1

    # 7. Sitting vs lying binary
    if ori.startswith('Seated'):
        feats['is_sitting'] = 1
        feats['is_lying'] = 0
    else:
        feats['is_sitting'] = 0
        feats['is_lying'] = 1

    return pd.Series(feats)


def build_features(interim_dir, demo_file, output_file, is_train=True):
    demo = pd.read_parquet(demo_file).set_index('subject')
    seq_files = [f for f in os.listdir(interim_dir) if f.endswith('.parquet') and 'demographics' not in f]
    all_feats = []
    for file in seq_files:
        df = pd.read_parquet(Path(interim_dir) / file)
        seq_id = file.replace('.parquet','')
        feats = extract_sequence_features(df)
        feats['sequence_id'] = seq_id
        feats['seq_length'] = len(df)
        feats['subject'] = df['subject'].iloc[0]
        if is_train:
            feats['gesture'] = df['gesture'].iloc[0]
            feats['sequence_type'] = df['sequence_type'].iloc[0]
        all_feats.append(feats)

    feat_df = pd.DataFrame(all_feats).set_index('sequence_id')
    # One-hot fill missing orientation dummies
    feat_df = feat_df.fillna(0)
    # Merge demographics
    feat_df = feat_df.merge(demo, left_on='subject', right_index=True, how='left')
    feat_df.to_csv(output_file)
    print(f'Features saved to {output_file}, shape {feat_df.shape}')


if __name__ == '__main__':
    build_features(INTERIM_DIR, INTERIM_DIR/'train_demographics.parquet', PROCESSED_DIR/'features_train.csv', True)
    td = INTERIM_DIR/'test_demographics.parquet'
    if td.exists():
        build_features(INTERIM_DIR, td, PROCESSED_DIR/'features_test.csv', False)
