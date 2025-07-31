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
    # vector magnitude for acc only
    if all(c in df.columns for c in ['acc_x','acc_y','acc_z']):
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        imu_axes.append('acc_mag')

    for c in imu_axes:
        arr = df[c].values
        stats = summary_stats(arr)
        for k, v in stats.items(): feats[f'{c}_{k}'] = v
        feats.update(fft_top_features(arr, f'{c}'))

    # 2. World-frame accel: linacc
    if all(c in df.columns for c in ['linacc_x','linacc_y','linacc_z']):
        df['linacc_mag'] = np.sqrt(df['linacc_x']**2 + df['linacc_y']**2 + df['linacc_z']**2)
        lin_axes = ['linacc_x','linacc_y','linacc_z','linacc_mag']
        # velocity (cumsum * dt)
        dt = 1.0 / SAMPLE_RATE
        linvel = np.cumsum(df[['linacc_x','linacc_y','linacc_z']].fillna(0), axis=0) * dt
        feats['linacc_disp_total'] = np.nansum(np.linalg.norm(linvel, axis=1))
        # kinetic energy approx
        feats['linacc_energy'] = np.nansum(0.5 * np.linalg.norm(linvel, axis=1)**2)
        for c in lin_axes:
            arr = df[c].values
            stats = summary_stats(arr)
            for k, v in stats.items(): feats[f'{c}_{k}'] = v
            feats.update(fft_top_features(arr, f'{c}'))

    # 3. Thermopiles
    thm_cols = [c for c in df.columns if c.startswith('thm_')]
    for c in thm_cols:
        arr = df[c].values
        stats = summary_stats(arr)
        feats.update({f'{c}_{k}':v for k,v in stats.items()})
        # diff
        diff = np.diff(arr)
        feats[f'{c}_diff_mean'] = np.nanmean(diff)
        feats[f'{c}_diff_std'] = np.nanstd(diff)
        # time to max
        feats[f'{c}_time_to_max'] = np.argmax(arr)

    # 4. Time-of-Flight sensors
    for i in range(1,6):
        tof_cols = [f'tof_{i}_v{j}' for j in range(64) if f'tof_{i}_v{j}' in df.columns]
        if not tof_cols: continue
        grid = df[tof_cols].fillna(-1).values
        flat = grid.flatten()
        stats = summary_stats(flat)
        feats.update({f'tof{i}_{k}':v for k,v in stats.items()})
        # histogram
        hist, _ = np.histogram(flat, bins=HIST_BINS, range=(flat.min(), flat.max()))
        hist = hist / (hist.sum()+1e-6)
        for b in range(HIST_BINS): feats[f'tof{i}_hist_{b}'] = hist[b]
        # simple texture: mean absolute neighbor diff
        diffs = np.abs(flat[1:] - flat[:-1])
        feats[f'tof{i}_texture'] = np.nanmean(diffs)

    # 5. Phase dynamics
    phase_counts = df['behavior'].value_counts().to_dict()
    total = len(df)
    for ph, cnt in phase_counts.items():
        key = ph.replace(' ', '_')
        feats[f'phase_{key}_cnt'] = cnt
        feats[f'phase_{key}_frac'] = cnt/total

    return pd.Series(feats)


def build_features(interim_dir, demo_file, output_file, is_train=True):
    # Load demographics
    demo = pd.read_parquet(demo_file)
    demo.set_index('subject', inplace=True)

    seq_files = [f for f in os.listdir(interim_dir) if f.endswith('.parquet') and not f.startswith('train_demographics')]
    all_feats = []
    for f in seq_files:
        df = pd.read_parquet(Path(interim_dir)/f)
        seq_id = f.replace('.parquet','')
        feats = extract_sequence_features(df)
        feats['sequence_id'] = seq_id
        # grab subject, gesture, type if train
        feats['subject'] = df['subject'].iloc[0]
        if is_train:
            feats['gesture'] = df['gesture'].iloc[0]
            feats['sequence_type'] = df['sequence_type'].iloc[0]
        all_feats.append(feats)

    feat_df = pd.DataFrame(all_feats).set_index('sequence_id')
    # merge demographics
    feat_df = feat_df.merge(demo, left_on='subject', right_index=True, how='left')

    feat_df.to_csv(output_file)
    print(f'Features saved to {output_file}, shape {feat_df.shape}')


if __name__ == '__main__':
    # train
    build_features(INTERIM_DIR, INTERIM_DIR/'train_demographics.parquet', PROCESSED_DIR/'features_train.csv', True)
    # test (if exists)
    test_demo = INTERIM_DIR/'test_demographics.parquet'
    if test_demo.exists():
        build_features(INTERIM_DIR, test_demo, PROCESSED_DIR/'features_test.csv', False)
