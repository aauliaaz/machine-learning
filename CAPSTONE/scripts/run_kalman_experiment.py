from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CAPSTONE_DIR = Path(__file__).resolve().parents[1]
if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))

from src import config as C
from src.data_loader import load_all_stations
from src.evaluation import compute_metrics
from src.feature_engineering import add_calendar_features
from src.model import build_lstm, set_seed, train_model
from src.preprocessing import build_pipeline, inverse_target


CALENDAR_COLS = [
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]

FEATURE_SETS = {
    "univariate": [C.TARGET],
    "target_calendar": CALENDAR_COLS + [C.TARGET],
    "multivariate": C.WEATHER_COLS + [C.AOD_COL, C.TARGET],
    "calendar": C.WEATHER_COLS + [C.AOD_COL] + CALENDAR_COLS + [C.TARGET],
}

SMOOTHING_VARIANTS: dict[str, dict[str, Any]] = {
    "no_smooth": {"smooth_aod": False},
    "rolling7": {
        "smooth_aod": True,
        "smooth_method": "rolling",
        "smooth_window": 7,
    },
    "kalman_slow": {
        "smooth_aod": True,
        "smooth_method": "kalman",
        "kalman_process_variance": 1e-4,
        "kalman_measurement_variance": 1e-2,
    },
    "kalman_fast": {
        "smooth_aod": True,
        "smooth_method": "kalman",
        "kalman_process_variance": 1e-3,
        "kalman_measurement_variance": 1e-2,
    },
}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_int_csv(value: str) -> list[int]:
    return [int(item) for item in _split_csv(value)]


def _run_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    """Stable identifier for one experiment combination."""
    return (
        str(row["station"]),
        str(row["feature_set"]),
        str(row["variant"]),
        int(row["seed"]),
        str(row["target_mode"]),
    )


def _load_existing_results(results_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, int, str]]]:
    if not results_path.exists():
        return [], set()

    existing = pd.read_csv(results_path)
    if existing.empty:
        return [], set()

    required = {"station", "feature_set", "variant", "seed", "target_mode"}
    if not required.issubset(existing.columns):
        raise ValueError(f"Existing results file is missing columns: {sorted(required - set(existing.columns))}")

    rows = existing.to_dict("records")
    completed = {_run_key(row) for row in rows if not row.get("error")}
    return rows, completed


def _load_existing_best(best_path: Path) -> dict[str, dict[str, Any]]:
    if not best_path.exists():
        return {}

    existing = pd.read_csv(best_path)
    if existing.empty or "station" not in existing.columns:
        return {}

    return {
        str(row["station"]): row
        for row in existing.to_dict("records")
    }


def prepare_station_frame(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Return dataframe and columns for a feature set."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set}")

    out = df.copy()
    if any(col in FEATURE_SETS[feature_set] for col in CALENDAR_COLS):
        out = add_calendar_features(out)
    return out, list(FEATURE_SETS[feature_set])


def _target_arrays(data: dict, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled target and persistence baseline for one split."""
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

    y_pred = inverse_target(
        y_pred_scaled,
        data["scaler"],
        data["target_idx"],
        n_features,
    )
    y_true = inverse_target(
        y_true_scaled,
        data["scaler"],
        data["target_idx"],
        n_features,
    )
    return compute_metrics(y_true, y_pred), y_true, y_pred


def _score_split(
    model,
    data: dict,
    split_name: str,
    target_mode: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Predict one split and return metrics on the original PM2.5 scale."""
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
    """Score y_t = y_(t-1) baseline on the original PM2.5 scale."""
    x_key = f"X_{split_name}"
    n_features = data[x_key].shape[2]
    y_true_scaled, y_persist_scaled = _target_arrays(data, split_name)
    metrics, _, _ = _score_prediction(y_true_scaled, y_persist_scaled, data, n_features)
    return metrics


def run_one(
    df: pd.DataFrame,
    station: str,
    feature_set: str,
    variant: str,
    seed: int,
    target_mode: str,
    model_params: dict[str, Any],
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[dict[str, Any], Any, pd.DataFrame | None]:
    """Train and evaluate one station-feature-smoothing-seed combination."""
    df_model, features = prepare_station_frame(df, feature_set)
    smooth_kwargs = SMOOTHING_VARIANTS[variant]

    data = build_pipeline(df_model, features, **smooth_kwargs)
    set_seed(seed)

    n_features = data["X_train"].shape[2]
    y_train_model, y_val_model = _training_targets(data, target_mode)

    model = build_lstm(input_shape=(C.LOOKBACK, n_features), **model_params)
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

    row = {
        "station": station,
        "feature_set": feature_set,
        "variant": variant,
        "seed": seed,
        "target_mode": target_mode,
        "n_features": n_features,
        "epochs_run": len(history.history["loss"]),
        "best_train_loss": float(np.min(history.history["loss"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"persistence_test_{k}": v for k, v in persistence_test.items()},
        **model_params,
    }

    pred_df = None
    if len(y_true) and len(y_pred):
        dates = data["splits"].test[C.DATE_COL].iloc[C.LOOKBACK:].reset_index(drop=True)
        pred_df = pd.DataFrame({
            C.DATE_COL: dates,
            "station": station,
            "feature_set": feature_set,
            "variant": variant,
            "seed": seed,
            "target_mode": target_mode,
            "actual_pm25": y_true,
            "predicted_pm25": y_pred,
        })

    return row, model, pred_df


def planned_runs(
    stations: list[str],
    feature_sets: list[str],
    variants: list[str],
    seeds: list[int],
    target_modes: list[str],
) -> list[tuple[str, str, str, int, str]]:
    """Build a run list and skip AOD smoothing for feature sets without AOD."""
    runs = []
    for station in stations:
        for feature_set in feature_sets:
            features = FEATURE_SETS[feature_set]
            for variant in variants:
                if C.AOD_COL not in features and variant != "no_smooth":
                    continue
                for seed in seeds:
                    for target_mode in target_modes:
                        runs.append((station, feature_set, variant, seed, target_mode))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", default=",".join(C.HEALTHY_STATIONS))
    parser.add_argument("--feature-sets", default="univariate,target_calendar,multivariate,calendar")
    parser.add_argument("--variants", default="no_smooth,rolling7,kalman_slow,kalman_fast")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--target-modes", default="level,residual")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip combinations that already exist in the results CSV.")
    args = parser.parse_args()

    stations = _split_csv(args.stations)
    feature_sets = _split_csv(args.feature_sets)
    variants = _split_csv(args.variants)
    seeds = _split_int_csv(args.seeds)
    target_modes = _split_csv(args.target_modes)

    unknown_stations = sorted(set(stations) - set(C.STATIONS))
    unknown_feature_sets = sorted(set(feature_sets) - set(FEATURE_SETS))
    unknown_variants = sorted(set(variants) - set(SMOOTHING_VARIANTS))
    unknown_target_modes = sorted(set(target_modes) - {"level", "residual"})
    if unknown_stations or unknown_feature_sets or unknown_variants or unknown_target_modes:
        raise ValueError(
            f"Unknown stations={unknown_stations}, "
            f"feature_sets={unknown_feature_sets}, variants={unknown_variants}, "
            f"target_modes={unknown_target_modes}"
        )

    results_path = C.METRICS_DIR / "11_kalman_experiment_results.csv"
    best_path = C.METRICS_DIR / "11_kalman_best_by_station.csv"

    runs = planned_runs(stations, feature_sets, variants, seeds, target_modes)
    existing_rows: list[dict[str, Any]] = []
    completed_runs: set[tuple[str, str, str, int, str]] = set()
    if args.resume:
        existing_rows, completed_runs = _load_existing_results(results_path)
        runs = [
            run
            for run in runs
            if run not in completed_runs
        ]

    if args.dry_run:
        if args.resume:
            print(f"Already completed runs: {len(completed_runs)}")
        print(f"Planned runs: {len(runs)}")
        print(pd.DataFrame(runs, columns=["station", "feature_set", "variant", "seed", "target_mode"]).head(20))
        return

    model_params = {
        "lstm_units": args.lstm_units,
        "dropout_rate": args.dropout_rate,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "activation": args.activation,
    }

    dfs = load_all_stations(reindex=True)
    rows: list[dict[str, Any]] = existing_rows.copy()
    best_by_station: dict[str, dict[str, Any]] = _load_existing_best(best_path) if args.resume else {}

    if args.resume:
        print(f"Resume mode: skipped {len(completed_runs)} completed runs, running {len(runs)} remaining runs.")

    for idx, (station, feature_set, variant, seed, target_mode) in enumerate(runs, 1):
        print(f"[{idx}/{len(runs)}] {station} | {feature_set} | {variant} | seed={seed} | {target_mode}")
        try:
            row, model, pred_df = run_one(
                dfs[station],
                station,
                feature_set,
                variant,
                seed,
                target_mode,
                model_params,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            row = {
                "station": station,
                "feature_set": feature_set,
                "variant": variant,
                "seed": seed,
                "target_mode": target_mode,
                "error": f"{type(exc).__name__}: {exc}",
                "val_R2": -1e6,
                "test_R2": -1e6,
            }
            model = None
            pred_df = None

        rows.append(row)

        current_best = best_by_station.get(station)
        if current_best is None or row.get("val_R2", -1e6) > current_best.get("val_R2", -1e6):
            best_by_station[station] = row.copy()
            if pred_df is not None:
                pred_path = C.METRICS_DIR / f"11_best_predictions_{station}.csv"
                pred_df.to_csv(pred_path, index=False)
                best_by_station[station]["prediction_path"] = str(pred_path)
            if model is not None and not args.no_save_models:
                model_path = C.MODEL_DIR / f"11_best_kalman_{station}.keras"
                model.save(model_path)
                best_by_station[station]["model_path"] = str(model_path)

        pd.DataFrame(rows).to_csv(results_path, index=False)
        pd.DataFrame(best_by_station.values()).to_csv(best_path, index=False)

    df_results = pd.DataFrame(rows).sort_values(["station", "val_R2"], ascending=[True, False])
    df_best = pd.DataFrame(best_by_station.values()).sort_values("test_R2", ascending=False)
    df_results.to_csv(results_path, index=False)
    df_best.to_csv(best_path, index=False)

    print("\nBest model per station, selected by validation R2:")
    print(df_best[[
        "station",
        "feature_set",
        "variant",
        "seed",
        "target_mode",
        "val_R2",
        "test_R2",
        "persistence_test_R2",
        "test_RMSE",
        "test_MAE",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
