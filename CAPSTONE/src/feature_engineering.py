from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config as C


# Lag features

def add_lag_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    lags: Iterable[int] = (1, 7, 14),
) -> pd.DataFrame:
    
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for k in lags:
            out[f"{col}_lag{k}"] = out[col].shift(k)
    return out


# Rolling statistics

def add_rolling_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    windows: Iterable[int] = (7, 14),
    stats: Iterable[str] = ("mean", "std"),
) -> pd.DataFrame:

    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s_lag = out[col].shift(1)  # exclude current value
        for w in windows:
            for stat in stats:
                if stat == "mean":
                    out[f"{col}_rmean{w}"] = s_lag.rolling(w, min_periods=1).mean()
                elif stat == "std":
                    out[f"{col}_rstd{w}"] = s_lag.rolling(w, min_periods=2).std()
                elif stat == "min":
                    out[f"{col}_rmin{w}"] = s_lag.rolling(w, min_periods=1).min()
                elif stat == "max":
                    out[f"{col}_rmax{w}"] = s_lag.rolling(w, min_periods=1).max()
                else:
                    raise ValueError(f"unknown stat: {stat}")
    return out


# Calendar features (cyclic encoding)

def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = C.DATE_COL,
) -> pd.DataFrame:
    """Tambah fitur kalender. Cyclic encoding (sin/cos) menghindari diskontinuitas
    di Senin↔Minggu dan Desember↔Januari.

    Polusi Jakarta sangat dipengaruhi pola weekly (lalu lintas weekend lebih
    rendah) dan monthly (musim hujan vs kemarau).
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])

    # Day of week: 0..6 → cyclic
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # Month: 1..12 → cyclic
    out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # Day of year (musim)
    out["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)

    # Weekend flag
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)

    return out


# ---------- Pipeline lengkap ----------

DEFAULT_LAG_COLS = (C.TARGET, C.AOD_COL)
DEFAULT_LAGS = (1, 7, 14)
DEFAULT_ROLLING_COLS = (C.TARGET, C.AOD_COL)
DEFAULT_ROLLING_WINDOWS = (7, 14)


def add_all_features(
    df: pd.DataFrame,
    lag_cols: Iterable[str] = DEFAULT_LAG_COLS,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_cols: Iterable[str] = DEFAULT_ROLLING_COLS,
    windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    add_calendar: bool = True,
    drop_initial_na: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply seluruh feature engineering. Return (df, daftar nama fitur baru).

    `drop_initial_na=True` membuang baris-baris awal yang NaN akibat lag/rolling.
    """
    out = df.copy()
    new_cols: list[str] = []

    out = add_lag_features(out, lag_cols, lags)
    new_cols += [f"{c}_lag{k}" for c in lag_cols if c in df.columns for k in lags]

    out = add_rolling_features(out, rolling_cols, windows)
    for c in rolling_cols:
        if c not in df.columns:
            continue
        for w in windows:
            new_cols += [f"{c}_rmean{w}", f"{c}_rstd{w}"]

    if add_calendar:
        out = add_calendar_features(out)
        new_cols += ["dow_sin", "dow_cos", "month_sin", "month_cos",
                     "doy_sin", "doy_cos", "is_weekend"]

    if drop_initial_na:
        # Buang baris awal yang NaN akibat lag terbesar
        max_lag = max(max(lags, default=0), max(windows, default=0))
        out = out.iloc[max_lag:].reset_index(drop=True)

    return out, new_cols
