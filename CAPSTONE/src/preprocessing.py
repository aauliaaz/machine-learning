from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config as C


# imputasi
DEFAULT_ZERO_AS_MISSING_COLS = (
    C.TARGET,
    C.AOD_COL,
    "temp",
    "dew",
    "humidity",
    "windspeed",
)


def replace_zero_with_nan(
    df: pd.DataFrame,
    cols: Sequence[str] = DEFAULT_ZERO_AS_MISSING_COLS,
) -> pd.DataFrame:

    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out.loc[out[col] == 0, col] = np.nan
    return out


def impute_linear_then_fill(
    df: pd.DataFrame,
    cols: Sequence[str],
    zero_as_missing_cols: Sequence[str] | None = DEFAULT_ZERO_AS_MISSING_COLS,
) -> pd.DataFrame:

    out = df.copy()
    if zero_as_missing_cols:
        out = replace_zero_with_nan(
            out,
            [col for col in zero_as_missing_cols if col in cols],
        )
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out[col].interpolate(method="linear", limit_direction="both")
        out[col] = out[col].ffill().bfill()
    return out


def kalman_filter_series(
    values: Sequence[float],
    process_variance: float = 1e-4,
    measurement_variance: float = 1e-2,
    initial_error: float = 1.0,
) -> np.ndarray:

    z = np.asarray(values, dtype=float)
    filtered = np.full_like(z, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(z))
    if len(finite_idx) == 0:
        return filtered

    start = int(finite_idx[0])
    x = float(z[start])
    p = float(initial_error)
    filtered[start] = x

    q = float(process_variance)
    r = float(measurement_variance)
    for i in range(start + 1, len(z)):
        x_pred = x
        p_pred = p + q

        if np.isfinite(z[i]):
            k = p_pred / (p_pred + r)
            x = x_pred + k * (float(z[i]) - x_pred)
            p = (1.0 - k) * p_pred
        else:
            x = x_pred
            p = p_pred
        filtered[i] = x

    return filtered


def causal_smooth_aod(
    df: pd.DataFrame,
    window: int = 7,
    method: Literal["rolling", "kalman"] = "rolling",
    kalman_process_variance: float = 1e-4,
    kalman_measurement_variance: float = 1e-2,
) -> pd.DataFrame:
    out = df.copy()
    if C.AOD_COL in out.columns:
        if method == "rolling":
            out[C.AOD_COL] = out[C.AOD_COL].rolling(window=window, min_periods=1).mean()
        elif method == "kalman":
            out[C.AOD_COL] = kalman_filter_series(
                out[C.AOD_COL].to_numpy(),
                process_variance=kalman_process_variance,
                measurement_variance=kalman_measurement_variance,
            )
        else:
            raise ValueError("method harus 'rolling' atau 'kalman'")
    return out


# split

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


# scaling

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


#  window time-series

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


# pipeline 

def build_pipeline(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str = C.TARGET,
    lookback: int = C.LOOKBACK,
    smooth_aod: bool = False,
    smooth_window: int = 7,
    smooth_method: Literal["rolling", "kalman"] = "rolling",
    zero_as_missing: bool = True,
    zero_as_missing_cols: Sequence[str] | None = None,
    kalman_process_variance: float = 1e-4,
    kalman_measurement_variance: float = 1e-2,
) -> dict:

    df_proc = df.copy()
    zero_cols = zero_as_missing_cols or DEFAULT_ZERO_AS_MISSING_COLS
    if zero_as_missing:
        df_proc = replace_zero_with_nan(df_proc, zero_cols)

    feature_cols = list(feature_cols)
    has_aod = C.AOD_COL in feature_cols

    if smooth_aod and has_aod and smooth_method == "kalman":
        non_aod_cols = [col for col in feature_cols if col != C.AOD_COL]
        df_proc = impute_linear_then_fill(
            df_proc,
            non_aod_cols,
            zero_as_missing_cols=(),
        )
        df_proc = causal_smooth_aod(
            df_proc,
            window=smooth_window,
            method=smooth_method,
            kalman_process_variance=kalman_process_variance,
            kalman_measurement_variance=kalman_measurement_variance,
        )
        df_proc = impute_linear_then_fill(
            df_proc,
            [C.AOD_COL],
            zero_as_missing_cols=(),
        )
    else:
        df_proc = impute_linear_then_fill(
            df_proc,
            feature_cols,
            zero_as_missing_cols=(),
        )
        if smooth_aod and has_aod:
            df_proc = causal_smooth_aod(
                df_proc,
                window=smooth_window,
                method=smooth_method,
                kalman_process_variance=kalman_process_variance,
                kalman_measurement_variance=kalman_measurement_variance,
            )
            df_proc = impute_linear_then_fill(
                df_proc,
                [C.AOD_COL],
                zero_as_missing_cols=(),
            )

    df_proc = df_proc.dropna(subset=feature_cols).reset_index(drop=True)

    splits = chronological_split(df_proc)
    scaler = fit_scaler_on_train(splits.train, feature_cols)

    tr = transform(splits.train, feature_cols, scaler)
    va = transform(splits.val, feature_cols, scaler)
    te = transform(splits.test, feature_cols, scaler)

    target_idx = list(feature_cols).index(target)

    X_tr, y_tr = make_sequences(tr, target_idx, lookback=lookback)
    X_va, y_va = make_sequences(va, target_idx, lookback=lookback)
    X_te, y_te = make_sequences(te, target_idx, lookback=lookback)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_va, "y_val":   y_va,
        "X_test":  X_te, "y_test":  y_te,
        "scaler":      scaler,
        "feature_cols": list(feature_cols),
        "target_idx":  target_idx,
        "lookback":    lookback,
        "df_processed": df_proc,
        "splits":      splits,
    }


def inverse_target(values: np.ndarray, scaler: MinMaxScaler, target_idx: int, n_features: int) -> np.ndarray:
    """Kembalikan target ke skala asli (µg/m³)."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]
