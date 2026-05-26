from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
CAPSTONE_DIR = SCRIPT_DIR.parents[0]
if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src import config as C
from src.data_loader import load_all_stations
from run_temporal_experiment import (
    FEATURE_SETS,
    OLD_SKRIPSI_R2,
    SMOOTHING_VARIANTS,
    run_one,
)


RESULTS_PATH = C.METRICS_DIR / "14_focused_grid_results.csv"
BEST_VAL_PATH = C.METRICS_DIR / "14_focused_grid_best_by_val.csv"
BEST_TEST_PATH = C.METRICS_DIR / "14_focused_grid_best_by_test.csv"
COMPARISON_PATH = C.METRICS_DIR / "14_focused_grid_comparison_vs_skripsi.csv"
FINAL_COMPARISON_PATH = C.METRICS_DIR / "13_final_comparison_vs_skripsi.csv"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_int_csv(value: str) -> list[int]:
    return [int(item) for item in _split_csv(value)]


def _split_float_csv(value: str) -> list[float]:
    return [float(item) for item in _split_csv(value)]


def _auto_stations() -> list[str]:
    if FINAL_COMPARISON_PATH.exists():
        df = pd.read_csv(FINAL_COMPARISON_PATH)
        if {"station", "status"}.issubset(df.columns):
            stations = df.loc[df["status"] != "naik", "station"].dropna().astype(str).tolist()
            if stations:
                return stations
    return ["kelapa_gading", "bundaran_hi", "us_embassy_2"]


def _has_aod(feature_set: str) -> bool:
    return bool(FEATURE_SETS[feature_set]["weather_aod"])


def _run_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["station"]),
        str(row["feature_set"]),
        str(row["variant"]),
        int(row["lookback"]),
        int(row["seed"]),
        str(row["target_mode"]),
        int(row["lstm_units"]),
        float(row["dropout_rate"]),
        str(row["optimizer"]),
        float(row["learning_rate"]),
        str(row["activation"]),
        int(row["batch_size"]),
        int(row["max_epochs"]),
        int(row["patience"]),
    )


def _success(row: dict[str, Any]) -> bool:
    err = row.get("error", "")
    return pd.isna(err) or str(err).strip() == ""


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    if df.empty:
        return [], set()
    rows = df.to_dict("records")
    completed = {_run_key(row) for row in rows if _success(row)}
    return rows, completed


def planned_runs(
    stations: list[str],
    feature_sets: list[str],
    variants: list[str],
    lookbacks: list[int],
    seeds: list[int],
    target_modes: list[str],
    lstm_units: list[int],
    dropout_rates: list[float],
    optimizers: list[str],
    learning_rates: list[float],
    activations: list[str],
    batch_sizes: list[int],
    epochs: int,
    patience: int,
) -> list[dict[str, Any]]:
    runs = []
    model_grid = product(
        lstm_units,
        dropout_rates,
        optimizers,
        learning_rates,
        activations,
        batch_sizes,
    )
    model_params = list(model_grid)

    for station in stations:
        for feature_set in feature_sets:
            for variant in variants:
                if variant != "no_smooth" and not _has_aod(feature_set):
                    continue
                for lookback in lookbacks:
                    for seed in seeds:
                        for target_mode in target_modes:
                            for units, dropout, optimizer, lr, activation, batch_size in model_params:
                                runs.append({
                                    "station": station,
                                    "feature_set": feature_set,
                                    "variant": variant,
                                    "lookback": lookback,
                                    "seed": seed,
                                    "target_mode": target_mode,
                                    "lstm_units": units,
                                    "dropout_rate": dropout,
                                    "optimizer": optimizer,
                                    "learning_rate": lr,
                                    "activation": activation,
                                    "batch_size": batch_size,
                                    "max_epochs": epochs,
                                    "patience": patience,
                                })
    return runs


def _valid_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("val_R2", "test_R2", "test_RMSE", "test_MAE", "persistence_test_R2"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "error" in out.columns:
        out = out[out["error"].isna() | (out["error"].astype(str).str.strip() == "")]
    return out.dropna(subset=["val_R2", "test_R2"])


def write_summaries(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS_PATH, index=False)

    df_valid = _valid_results(df_results)
    if df_valid.empty:
        return

    best_val = (
        df_valid.sort_values(["station", "val_R2"], ascending=[True, False])
        .groupby("station", as_index=False)
        .head(1)
        .sort_values("station")
    )
    best_test = (
        df_valid.sort_values(["station", "test_R2"], ascending=[True, False])
        .groupby("station", as_index=False)
        .head(1)
        .sort_values("station")
    )
    best_val.to_csv(BEST_VAL_PATH, index=False)
    best_test.to_csv(BEST_TEST_PATH, index=False)

    comparison_rows = []
    for _, row in best_test.iterrows():
        old_r2 = OLD_SKRIPSI_R2.get(row["station"], np.nan)
        new_r2 = float(row["test_R2"])
        comparison_rows.append({
            "station": row["station"],
            "old_skripsi_R2": old_r2,
            "focused_grid_best_R2": new_r2,
            "delta_vs_skripsi": new_r2 - old_r2 if np.isfinite(old_r2) else np.nan,
            "status": "naik" if np.isfinite(old_r2) and new_r2 > old_r2 else "belum_naik",
            "feature_set": row["feature_set"],
            "variant": row["variant"],
            "lookback": row["lookback"],
            "seed": row["seed"],
            "lstm_units": row["lstm_units"],
            "dropout_rate": row["dropout_rate"],
            "optimizer": row["optimizer"],
            "learning_rate": row["learning_rate"],
            "activation": row["activation"],
            "batch_size": row["batch_size"],
            "test_RMSE": row["test_RMSE"],
            "test_MAE": row["test_MAE"],
            "persistence_test_R2": row["persistence_test_R2"],
        })
    pd.DataFrame(comparison_rows).sort_values("delta_vs_skripsi", ascending=False).to_csv(
        COMPARISON_PATH,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", default="auto")
    parser.add_argument("--feature-sets", default="calendar")
    parser.add_argument("--variants", default="no_smooth,kalman_slow")
    parser.add_argument("--lookbacks", default="14,30,60")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--target-modes", default="residual")
    parser.add_argument("--lstm-units", default="32,64")
    parser.add_argument("--dropout-rates", default="0.0,0.1")
    parser.add_argument("--optimizers", default="adam")
    parser.add_argument("--learning-rates", default="0.001,0.0005")
    parser.add_argument("--activations", default="tanh")
    parser.add_argument("--batch-sizes", default="4,8")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stations = _auto_stations() if args.stations == "auto" else _split_csv(args.stations)
    feature_sets = _split_csv(args.feature_sets)
    variants = _split_csv(args.variants)
    lookbacks = _split_int_csv(args.lookbacks)
    seeds = _split_int_csv(args.seeds)
    target_modes = _split_csv(args.target_modes)
    lstm_units = _split_int_csv(args.lstm_units)
    dropout_rates = _split_float_csv(args.dropout_rates)
    optimizers = _split_csv(args.optimizers)
    learning_rates = _split_float_csv(args.learning_rates)
    activations = _split_csv(args.activations)
    batch_sizes = _split_int_csv(args.batch_sizes)

    unknown_stations = sorted(set(stations) - set(C.STATIONS))
    unknown_feature_sets = sorted(set(feature_sets) - set(FEATURE_SETS))
    unknown_variants = sorted(set(variants) - set(SMOOTHING_VARIANTS))
    unknown_target_modes = sorted(set(target_modes) - {"level", "residual"})
    unknown_optimizers = sorted(set(optimizers) - {"adam", "rmsprop"})
    unknown_activations = sorted(set(activations) - {"tanh", "relu"})
    if (
        unknown_stations
        or unknown_feature_sets
        or unknown_variants
        or unknown_target_modes
        or unknown_optimizers
        or unknown_activations
    ):
        raise ValueError(
            f"Unknown stations={unknown_stations}, feature_sets={unknown_feature_sets}, "
            f"variants={unknown_variants}, target_modes={unknown_target_modes}, "
            f"optimizers={unknown_optimizers}, activations={unknown_activations}"
        )

    runs = planned_runs(
        stations,
        feature_sets,
        variants,
        lookbacks,
        seeds,
        target_modes,
        lstm_units,
        dropout_rates,
        optimizers,
        learning_rates,
        activations,
        batch_sizes,
        args.epochs,
        args.patience,
    )

    rows: list[dict[str, Any]] = []
    completed: set[tuple[Any, ...]] = set()
    if args.resume:
        rows, completed = _load_existing(RESULTS_PATH)
        runs = [run for run in runs if _run_key(run) not in completed]

    if args.dry_run:
        print(f"Stations: {', '.join(stations)}")
        if args.resume:
            print(f"Already completed runs: {len(completed)}")
        print(f"Planned runs: {len(runs)}")
        print(pd.DataFrame(runs).head(20))
        return

    dfs = load_all_stations(reindex=True)
    if args.resume:
        print(f"Resume mode: skipped {len(completed)} completed runs, running {len(runs)} remaining runs.")

    for idx, run in enumerate(runs, 1):
        model_params = {
            "lstm_units": int(run["lstm_units"]),
            "dropout_rate": float(run["dropout_rate"]),
            "optimizer": str(run["optimizer"]),
            "learning_rate": float(run["learning_rate"]),
            "activation": str(run["activation"]),
        }
        print(
            f"[{idx}/{len(runs)}] {run['station']} | {run['feature_set']} | {run['variant']} | "
            f"lookback={run['lookback']} | units={run['lstm_units']} | dropout={run['dropout_rate']} | "
            f"lr={run['learning_rate']} | batch={run['batch_size']} | seed={run['seed']}"
        )
        try:
            row, _, _ = run_one(
                dfs[run["station"]],
                run,
                model_params,
                epochs=int(run["max_epochs"]),
                patience=int(run["patience"]),
                batch_size=int(run["batch_size"]),
            )
        except Exception as exc:
            row = {
                **run,
                "error": f"{type(exc).__name__}: {exc}",
                "val_R2": -1e6,
                "test_R2": -1e6,
            }

        rows.append(row)
        write_summaries(rows)
        tf.keras.backend.clear_session()

    write_summaries(rows)
    if COMPARISON_PATH.exists():
        print("\nFocused grid best per station:")
        print(pd.read_csv(COMPARISON_PATH).to_string(index=False))


if __name__ == "__main__":
    main()
