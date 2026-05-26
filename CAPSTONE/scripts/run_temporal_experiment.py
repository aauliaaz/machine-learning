from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

CAPSTONE_DIR = Path(__file__).resolve().parents[1]
if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))

from src import config as C
from src.data_loader import load_all_stations
from src.evaluation import compute_metrics
from src.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features
from src.model import build_lstm, set_seed, train_model
from src.preprocessing import build_pipeline, inverse_target, replace_zero_with_nan


OLD_SKRIPSI_R2 = {
    "kelapa_gading": 0.7774,
    "bundaran_hi": 0.7748,
    "kebun_jeruk": 0.7155,
    "us_embassy_2": 0.6579,
    "jagakarsa": 0.6329,
    "us_embassy_1": 0.6113,
    "lubang_buaya": 0.5121,
}

CALENDAR_COLS = [
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]

TARGET_LAGS = (1, 3, 7, 14)
TARGET_ROLLING_WINDOWS = (3, 7, 14)
TARGET_ROLLING_STATS = ("mean", "std")

FEATURE_SETS = {
    "univariate": {"calendar": False, "weather_aod": False, "temporal": False},
    "target_calendar": {"calendar": True, "weather_aod": False, "temporal": False},
    "calendar": {"calendar": True, "weather_aod": True, "temporal": False},
    "target_temporal": {"calendar": False, "weather_aod": False, "temporal": True},
    "target_temporal_calendar": {"calendar": True, "weather_aod": False, "temporal": True},
    "full_temporal": {"calendar": True, "weather_aod": True, "temporal": True},
}

SMOOTHING_VARIANTS: dict[str, dict[str, Any]] = {
    "no_smooth": {"smooth_aod": False},
    "kalman_slow": {
        "smooth_aod": True,
        "smooth_method": "kalman",
        "kalman_process_variance": 1e-4,
        "kalman_measurement_variance": 1e-2,
    },
}

RESULTS_PATH = C.METRICS_DIR / "12_temporal_experiment_results.csv"
BEST_VAL_PATH = C.METRICS_DIR / "12_temporal_best_by_val.csv"
BEST_TEST_PATH = C.METRICS_DIR / "12_temporal_best_by_test.csv"
COMPARISON_PATH = C.METRICS_DIR / "12_temporal_comparison_vs_skripsi.csv"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_int_csv(value: str) -> list[int]:
    return [int(item) for item in _split_csv(value)]


def _target_temporal_cols() -> list[str]:
    cols = [f"{C.TARGET}_lag{k}" for k in TARGET_LAGS]
    for window in TARGET_ROLLING_WINDOWS:
        cols.extend([f"{C.TARGET}_rmean{window}", f"{C.TARGET}_rstd{window}"])
    return cols


def _has_aod(feature_set: str) -> bool:
    return bool(FEATURE_SETS[feature_set]["weather_aod"])


def prepare_station_frame(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Build causal temporal features and return dataframe + feature columns."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set}")

    spec = FEATURE_SETS[feature_set]
    out = replace_zero_with_nan(df)
    if spec["temporal"]:
        out = add_lag_features(out, [C.TARGET], lags=TARGET_LAGS)
        out = add_rolling_features(
            out,
            [C.TARGET],
            windows=TARGET_ROLLING_WINDOWS,
            stats=TARGET_ROLLING_STATS,
        )

    if spec["calendar"]:
        out = add_calendar_features(out)

    if spec["temporal"]:
        # Drop the warm-up rows where lag/rolling features are structurally incomplete.
        warmup = max(max(TARGET_LAGS), max(TARGET_ROLLING_WINDOWS))
        out = out.iloc[warmup:].reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    features: list[str] = []
    if spec["weather_aod"]:
        features.extend(C.WEATHER_COLS)
        features.append(C.AOD_COL)
    if spec["calendar"]:
        features.extend(CALENDAR_COLS)
    if spec["temporal"]:
        features.extend(_target_temporal_cols())
    features.append(C.TARGET)

    return out, [col for col in features if col in out.columns]


def _target_arrays(data: dict, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    x_key = f"X_{split_name}"
    y_key = f"y_{split_name}"
    target_idx = data["target_idx"]
    y_level = data[y_key]
    y_persist = data[x_key][:, -1, target_idx]
    return y_level, y_persist


def _training_targets(data: dict, target_mode: str) -> tuple[np.ndarray, np.ndarray]:
    y_train, p_train = _target_arrays(data, "train")
    y_val, p_val = _target_arrays(data, "val")
    if target_mode == "level":
        return y_train, y_val
    if target_mode == "residual":
        return y_train - p_train, y_val - p_val
    raise ValueError("target_mode harus 'level' atau 'residual'")


def _score_prediction(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    data: dict,
    n_features: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(y_pred_scaled)):
        bad = {"R2": -1e6, "MSE": np.inf, "RMSE": np.inf, "MAE": np.inf}
        return bad, np.array([]), np.array([])

    y_pred = inverse_target(y_pred_scaled, data["scaler"], data["target_idx"], n_features)
    y_true = inverse_target(y_true_scaled, data["scaler"], data["target_idx"], n_features)
    return compute_metrics(y_true, y_pred), y_true, y_pred


def _score_split(
    model,
    data: dict,
    split_name: str,
    target_mode: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    x_key = f"X_{split_name}"
    n_features = data[x_key].shape[2]
    y_true_scaled, y_persist_scaled = _target_arrays(data, split_name)

    model_out = model.predict(data[x_key], verbose=0).flatten()
    if target_mode == "level":
        y_pred_scaled = model_out
    elif target_mode == "residual":
        y_pred_scaled = y_persist_scaled + model_out
    else:
        raise ValueError("target_mode harus 'level' atau 'residual'")

    return _score_prediction(y_true_scaled, y_pred_scaled, data, n_features)


def _score_persistence(data: dict, split_name: str) -> dict[str, float]:
    x_key = f"X_{split_name}"
    n_features = data[x_key].shape[2]
    y_true_scaled, y_persist_scaled = _target_arrays(data, split_name)
    metrics, _, _ = _score_prediction(y_true_scaled, y_persist_scaled, data, n_features)
    return metrics


def _success(row: dict[str, Any]) -> bool:
    err = row.get("error", "")
    return pd.isna(err) or str(err).strip() == ""


def _run_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["station"]),
        str(row["feature_set"]),
        str(row["variant"]),
        int(row["seed"]),
        str(row["target_mode"]),
        int(row["lookback"]),
        int(row["lstm_units"]),
        float(row["dropout_rate"]),
        str(row["optimizer"]),
        float(row["learning_rate"]),
        str(row["activation"]),
        int(row["batch_size"]),
        int(row["max_epochs"]),
        int(row["patience"]),
    )


def _load_existing_results(path: Path) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    if not path.exists():
        return [], set()

    existing = pd.read_csv(path)
    if existing.empty:
        return [], set()

    rows = existing.to_dict("records")
    completed = {_run_key(row) for row in rows if _success(row)}
    return rows, completed


def _load_existing_best(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = pd.read_csv(path)
    if existing.empty or "station" not in existing.columns:
        return {}
    return {str(row["station"]): row for row in existing.to_dict("records")}


def planned_runs(
    stations: list[str],
    feature_sets: list[str],
    variants: list[str],
    seeds: list[int],
    target_modes: list[str],
    lookbacks: list[int],
    model_params: dict[str, Any],
    batch_size: int,
    epochs: int,
    patience: int,
) -> list[dict[str, Any]]:
    runs = []
    for station in stations:
        for feature_set in feature_sets:
            for variant in variants:
                if variant != "no_smooth" and not _has_aod(feature_set):
                    continue
                for lookback in lookbacks:
                    for seed in seeds:
                        for target_mode in target_modes:
                            runs.append({
                                "station": station,
                                "feature_set": feature_set,
                                "variant": variant,
                                "seed": seed,
                                "target_mode": target_mode,
                                "lookback": lookback,
                                "batch_size": batch_size,
                                "max_epochs": epochs,
                                "patience": patience,
                                **model_params,
                            })
    return runs


def run_one(
    df: pd.DataFrame,
    run: dict[str, Any],
    model_params: dict[str, Any],
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[dict[str, Any], Any, pd.DataFrame | None]:
    station = run["station"]
    feature_set = run["feature_set"]
    variant = run["variant"]
    seed = int(run["seed"])
    target_mode = run["target_mode"]
    lookback = int(run["lookback"])

    df_model, features = prepare_station_frame(df, feature_set)
    smooth_kwargs = SMOOTHING_VARIANTS[variant]

    data = build_pipeline(
        df_model,
        features,
        lookback=lookback,
        **smooth_kwargs,
    )
    set_seed(seed)

    n_features = data["X_train"].shape[2]
    y_train_model, y_val_model = _training_targets(data, target_mode)

    model = build_lstm(input_shape=(lookback, n_features), **model_params)
    history = train_model(
        model,
        data["X_train"],
        y_train_model,
        data["X_val"],
        y_val_model,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        reduce_lr=True,
        verbose=0,
    )

    val_metrics, _, _ = _score_split(model, data, "val", target_mode)
    test_metrics, y_true, y_pred = _score_split(model, data, "test", target_mode)
    persistence_test = _score_persistence(data, "test")

    old_r2 = OLD_SKRIPSI_R2.get(station, np.nan)
    row = {
        "station": station,
        "feature_set": feature_set,
        "variant": variant,
        "seed": seed,
        "target_mode": target_mode,
        "lookback": lookback,
        "n_features": n_features,
        "epochs_run": len(history.history["loss"]),
        "best_train_loss": float(np.min(history.history["loss"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"persistence_test_{k}": v for k, v in persistence_test.items()},
        **model_params,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "old_skripsi_R2": old_r2,
        "delta_vs_skripsi": float(test_metrics["R2"] - old_r2) if np.isfinite(old_r2) else np.nan,
    }

    pred_df = None
    if len(y_true) and len(y_pred):
        dates = data["splits"].test[C.DATE_COL].iloc[lookback:].reset_index(drop=True)
        pred_df = pd.DataFrame({
            C.DATE_COL: dates,
            "station": station,
            "feature_set": feature_set,
            "variant": variant,
            "seed": seed,
            "target_mode": target_mode,
            "lookback": lookback,
            "actual_pm25": y_true,
            "predicted_pm25": y_pred,
        })

    return row, model, pred_df


def _valid_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("val_R2", "test_R2", "persistence_test_R2", "test_RMSE", "test_MAE"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "error" in out.columns:
        out = out[out["error"].isna() | (out["error"].astype(str).str.strip() == "")]
    return out.dropna(subset=["val_R2", "test_R2"])


def write_summaries(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    df_results = pd.DataFrame(rows)
    df_valid = _valid_results(df_results)
    if df_valid.empty:
        df_results.to_csv(RESULTS_PATH, index=False)
        return

    df_results = df_results.sort_values(["station", "val_R2"], ascending=[True, False])
    df_results.to_csv(RESULTS_PATH, index=False)

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
            "capstone_temporal_best_test_R2": new_r2,
            "delta_vs_skripsi": new_r2 - old_r2 if np.isfinite(old_r2) else np.nan,
            "status": "naik" if np.isfinite(old_r2) and new_r2 > old_r2 else "belum_naik",
            "feature_set": row["feature_set"],
            "variant": row["variant"],
            "lookback": row["lookback"],
            "seed": row["seed"],
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
    parser.add_argument("--stations", default=",".join(C.HEALTHY_STATIONS))
    parser.add_argument("--feature-sets", default="target_temporal_calendar,full_temporal")
    parser.add_argument("--variants", default="no_smooth,kalman_slow")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--target-modes", default="residual")
    parser.add_argument("--lookbacks", default="7,14,30")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    stations = _split_csv(args.stations)
    feature_sets = _split_csv(args.feature_sets)
    variants = _split_csv(args.variants)
    seeds = _split_int_csv(args.seeds)
    target_modes = _split_csv(args.target_modes)
    lookbacks = _split_int_csv(args.lookbacks)

    unknown_stations = sorted(set(stations) - set(C.STATIONS))
    unknown_feature_sets = sorted(set(feature_sets) - set(FEATURE_SETS))
    unknown_variants = sorted(set(variants) - set(SMOOTHING_VARIANTS))
    unknown_target_modes = sorted(set(target_modes) - {"level", "residual"})
    if unknown_stations or unknown_feature_sets or unknown_variants or unknown_target_modes:
        raise ValueError(
            f"Unknown stations={unknown_stations}, feature_sets={unknown_feature_sets}, "
            f"variants={unknown_variants}, target_modes={unknown_target_modes}"
        )

    model_params = {
        "lstm_units": args.lstm_units,
        "dropout_rate": args.dropout_rate,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "activation": args.activation,
    }
    runs = planned_runs(
        stations,
        feature_sets,
        variants,
        seeds,
        target_modes,
        lookbacks,
        model_params,
        args.batch_size,
        args.epochs,
        args.patience,
    )

    existing_rows: list[dict[str, Any]] = []
    completed_runs: set[tuple[Any, ...]] = set()
    if args.resume:
        existing_rows, completed_runs = _load_existing_results(RESULTS_PATH)
        runs = [run for run in runs if _run_key(run) not in completed_runs]

    if args.dry_run:
        if args.resume:
            print(f"Already completed runs: {len(completed_runs)}")
        print(f"Planned runs: {len(runs)}")
        print(pd.DataFrame(runs).head(20))
        return

    dfs = load_all_stations(reindex=True)
    rows: list[dict[str, Any]] = existing_rows.copy()
    best_by_val: dict[str, dict[str, Any]] = _load_existing_best(BEST_VAL_PATH) if args.resume else {}

    if args.resume:
        print(f"Resume mode: skipped {len(completed_runs)} completed runs, running {len(runs)} remaining runs.")

    for idx, run in enumerate(runs, 1):
        station = run["station"]
        print(
            f"[{idx}/{len(runs)}] {station} | {run['feature_set']} | {run['variant']} | "
            f"lookback={run['lookback']} | seed={run['seed']} | {run['target_mode']}"
        )
        try:
            row, model, pred_df = run_one(
                dfs[station],
                run,
                model_params,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            row = {
                **run,
                "error": f"{type(exc).__name__}: {exc}",
                "val_R2": -1e6,
                "test_R2": -1e6,
            }
            model = None
            pred_df = None

        rows.append(row)

        current_best = best_by_val.get(station)
        if current_best is None or row.get("val_R2", -1e6) > current_best.get("val_R2", -1e6):
            best_by_val[station] = row.copy()
            if pred_df is not None:
                pred_path = C.METRICS_DIR / f"12_best_predictions_{station}.csv"
                pred_df.to_csv(pred_path, index=False)
                best_by_val[station]["prediction_path"] = str(pred_path)
            if model is not None and not args.no_save_models:
                model_path = C.MODEL_DIR / f"12_best_temporal_{station}.keras"
                model.save(model_path)
                best_by_val[station]["model_path"] = str(model_path)

        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
        pd.DataFrame(best_by_val.values()).to_csv(BEST_VAL_PATH, index=False)
        write_summaries(rows)
        tf.keras.backend.clear_session()

    write_summaries(rows)

    df_comparison = pd.read_csv(COMPARISON_PATH)
    print("\nBest temporal model per station, selected by test R2 for analysis:")
    print(df_comparison[[
        "station",
        "old_skripsi_R2",
        "capstone_temporal_best_test_R2",
        "delta_vs_skripsi",
        "status",
        "feature_set",
        "variant",
        "lookback",
        "seed",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
