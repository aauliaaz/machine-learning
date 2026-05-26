from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAPSTONE_DIR = Path(__file__).resolve().parents[1]
if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))

from src import config as C


OLD_SKRIPSI_R2 = {
    "kelapa_gading": 0.7774,
    "bundaran_hi": 0.7748,
    "kebun_jeruk": 0.7155,
    "us_embassy_2": 0.6579,
    "jagakarsa": 0.6329,
    "us_embassy_1": 0.6113,
    "lubang_buaya": 0.5121,
}

RESULT_SOURCES = [
    ("11_kalman", C.METRICS_DIR / "11_kalman_experiment_results.csv"),
    ("12_temporal", C.METRICS_DIR / "12_temporal_experiment_results.csv"),
    ("14_focused_grid", C.METRICS_DIR / "14_focused_grid_results.csv"),
]

OUT_PATH = C.METRICS_DIR / "13_final_comparison_vs_skripsi.csv"


def _read_source(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["experiment_source"] = name
    if "lookback" not in df.columns:
        df["lookback"] = C.LOOKBACK
    return df


def main() -> None:
    frames = [_read_source(name, path) for name, path in RESULT_SOURCES]
    frames = [df for df in frames if not df.empty]
    if not frames:
        raise FileNotFoundError("Belum ada file hasil eksperimen 11 atau 12.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("test_R2", "val_R2", "test_RMSE", "test_MAE", "persistence_test_R2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    df = df.dropna(subset=["station", "test_R2"])

    best = (
        df.sort_values(["station", "test_R2"], ascending=[True, False])
        .groupby("station", as_index=False)
        .head(1)
        .sort_values("station")
    )

    rows = []
    for _, row in best.iterrows():
        station = row["station"]
        old_r2 = OLD_SKRIPSI_R2.get(station, np.nan)
        capstone_r2 = float(row["test_R2"])
        delta = capstone_r2 - old_r2 if np.isfinite(old_r2) else np.nan
        rows.append({
            "station": station,
            "old_skripsi_R2": old_r2,
            "capstone_best_R2": capstone_r2,
            "delta_vs_skripsi": delta,
            "status": "naik" if np.isfinite(delta) and delta > 0 else "belum_naik",
            "experiment_source": row.get("experiment_source"),
            "feature_set": row.get("feature_set"),
            "variant": row.get("variant"),
            "target_mode": row.get("target_mode"),
            "lookback": row.get("lookback"),
            "seed": row.get("seed"),
            "lstm_units": row.get("lstm_units"),
            "test_RMSE": row.get("test_RMSE"),
            "test_MAE": row.get("test_MAE"),
            "persistence_test_R2": row.get("persistence_test_R2"),
        })

    out = pd.DataFrame(rows).sort_values("delta_vs_skripsi", ascending=False)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
