
from __future__ import annotations

import pandas as pd

from . import config as C


def load_station(station: str) -> pd.DataFrame:
   
    if station not in C.STATIONS:
        raise KeyError(f"Stasiun '{station}' tidak dikenal. Pilih: {list(C.STATIONS)}")

    path = C.RAW_MERGED_DIR / f"{station}.csv"
    df = pd.read_csv(path)
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    df = (
        df.sort_values(C.DATE_COL)
          .drop_duplicates(subset=C.DATE_COL, keep="first")
          .reset_index(drop=True)
    )
    return df


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
 
    full_idx = pd.date_range(df[C.DATE_COL].min(), df[C.DATE_COL].max(), freq="D")
    out = (
        df.set_index(C.DATE_COL)
          .reindex(full_idx)
          .rename_axis(C.DATE_COL)
          .reset_index()
    )
    return out


def load_all_stations(reindex: bool = True) -> dict[str, pd.DataFrame]:
    out = {}
    for s in C.STATIONS:
        df = load_station(s)
        if reindex:
            df = reindex_daily(df)
        out[s] = df
    return out


def missing_value_report(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ringkasan missing value per stasiun per kolom."""
    rows = []
    for name, df in dfs.items():
        rec = {"station": name, "n_rows": len(df)}
        for col in [C.TARGET, C.AOD_COL, *C.WEATHER_COLS]:
            if col in df.columns:
                rec[f"missing_{col}"] = int(df[col].isna().sum())
        rows.append(rec)
    return pd.DataFrame(rows)
