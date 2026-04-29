"""Praproses: imputasi → split → scaling.

Memperbaiki dua masalah utama dari kode skripsi:
1. Scaler **fit hanya di data train** untuk mencegah leakage.
2. Smoothing AOD bersifat causal (tidak menggunakan data masa depan).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config as C


# ---------- imputasi ----------

def impute_linear_then_fill(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    """Interpolasi linear lalu ffill+bfill untuk mengisi sisa NaN di tepi."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out[col].interpolate(method="linear", limit_direction="both")
        out[col] = out[col].ffill().bfill()
    return out


def causal_smooth_aod(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling mean **tanpa** `center=True` agar tidak menggunakan masa depan."""
    out = df.copy()
    if C.AOD_COL in out.columns:
        out[C.AOD_COL] = out[C.AOD_COL].rolling(window=window, min_periods=1).mean()
    return out


# ---------- split ----------

@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = C.TRAIN_RATIO,
    val_ratio: float = C.VAL_RATIO,
) -> Splits:
    n = len(df)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return Splits(
        train=df.iloc[:n_tr].reset_index(drop=True),
        val=df.iloc[n_tr:n_tr + n_va].reset_index(drop=True),
        test=df.iloc[n_tr + n_va:].reset_index(drop=True),
    )


# ---------- scaling ----------

def fit_scaler_on_train(
    train: pd.DataFrame,
    feature_cols: Sequence[str],
) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train[list(feature_cols)].values)
    return scaler


def transform(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scaler: MinMaxScaler,
) -> np.ndarray:
    return scaler.transform(df[list(feature_cols)].values)


# ---------- pembuatan window time-series ----------

def make_sequences(
    arr: np.ndarray,
    target_idx: int,
    lookback: int = C.LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Mengubah array 2D (T, F) menjadi (X: samples×lookback×F, y: samples)."""
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i])
        y.append(arr[i, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ---------- pipeline lengkap ----------

def build_pipeline(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str = C.TARGET,
    smooth_aod: bool = False,
    smooth_window: int = 7,
) -> dict:
    """End-to-end: imputasi → split → scaling (fit on train) → sequences.

    Mengembalikan dict berisi X/y untuk train/val/test, scaler, dan kolom fitur.
    """
    df_proc = impute_linear_then_fill(df, list(feature_cols))
    if smooth_aod and C.AOD_COL in feature_cols:
        df_proc = causal_smooth_aod(df_proc, window=smooth_window)
        df_proc = impute_linear_then_fill(df_proc, [C.AOD_COL])

    df_proc = df_proc.dropna(subset=list(feature_cols)).reset_index(drop=True)

    splits = chronological_split(df_proc)
    scaler = fit_scaler_on_train(splits.train, feature_cols)

    tr = transform(splits.train, feature_cols, scaler)
    va = transform(splits.val, feature_cols, scaler)
    te = transform(splits.test, feature_cols, scaler)

    target_idx = list(feature_cols).index(target)

    X_tr, y_tr = make_sequences(tr, target_idx)
    X_va, y_va = make_sequences(va, target_idx)
    X_te, y_te = make_sequences(te, target_idx)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_va, "y_val":   y_va,
        "X_test":  X_te, "y_test":  y_te,
        "scaler":      scaler,
        "feature_cols": list(feature_cols),
        "target_idx":  target_idx,
        "df_processed": df_proc,
        "splits":      splits,
    }


def inverse_target(values: np.ndarray, scaler: MinMaxScaler, target_idx: int, n_features: int) -> np.ndarray:
    """Kembalikan target ke skala asli (µg/m³)."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]
