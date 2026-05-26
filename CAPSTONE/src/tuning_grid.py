from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from . import config as C
from .evaluation import compute_metrics
from .model import build_lstm, set_seed, train_model
from .preprocessing import build_pipeline, inverse_target


DEFAULT_GRID = {
    "learning_rate": [1e-3, 1e-2],
    "optimizer": ["adam", "rmsprop"],
    "lstm_units": [16, 32, 64],
    "dropout_rate": [0.0, 0.1, 0.2],
}


def _score_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scaler,
    target_idx: int,
    n_features: int,
    inverse_scale: bool = True,
) -> dict:

    y_pred = model.predict(X, verbose=0).flatten()
    y_true = y

    if not np.all(np.isfinite(y_pred)):
        return {"R2": -1e6, "MSE": np.inf, "RMSE": np.inf, "MAE": np.inf}

    if inverse_scale:
        y_pred = inverse_target(y_pred, scaler, target_idx, n_features)
        y_true = inverse_target(y_true, scaler, target_idx, n_features)

    return compute_metrics(y_true, y_pred)


def _is_higher_better(metric_name: str) -> bool:
    return metric_name.upper().endswith("R2")


def grid_search(
    data: dict,
    grid: dict | None = None,
    epochs: int = 100,
    batch_size: int = 4,
    patience: int = 15,
    inverse_scale: bool = True,
    selection_metric: str = "val_R2",
    verbose: int = 0,
) -> tuple[pd.DataFrame, dict]:

    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    rows: list[dict] = []
    n_features = data["X_train"].shape[2]
    maximize = _is_higher_better(selection_metric)
    best_score = -np.inf if maximize else np.inf
    best = {
        "score": best_score,
        "R2": None,
        "val_R2": None,
        "test_R2": None,
        "model": None,
        "params": None,
        "history": None,
        "val_metrics": None,
        "test_metrics": None,
        "epochs_run": None,
        "selection_metric": selection_metric,
    }

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        if verbose:
            print(f"[{i}/{len(combos)}] {params}")

        set_seed()
        model = build_lstm(input_shape=(C.LOOKBACK, n_features), **params)
        history = train_model(
            model,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=0,
        )

        val_metrics = _score_split(
            model,
            data["X_val"],
            data["y_val"],
            data["scaler"],
            data["target_idx"],
            n_features,
            inverse_scale=inverse_scale,
        )
        test_metrics = _score_split(
            model,
            data["X_test"],
            data["y_test"],
            data["scaler"],
            data["target_idx"],
            n_features,
            inverse_scale=inverse_scale,
        )

        row = {
            **params,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "epochs_run": len(history.history["loss"]),
            "best_train_loss": float(np.min(history.history["loss"])),
            "best_val_loss": float(np.min(history.history["val_loss"])),
        }
        rows.append(row)

        if selection_metric not in row:
            raise KeyError(
                f"selection_metric '{selection_metric}' tidak ditemukan. "
                "Gunakan kolom hasil seperti val_R2, val_RMSE, test_R2, atau test_RMSE."
            )

        score = row[selection_metric]
        is_better = score > best_score if maximize else score < best_score
        if is_better:
            best_score = score
            best = {
                "score": score,
                "R2": test_metrics["R2"],
                "val_R2": val_metrics["R2"],
                "test_R2": test_metrics["R2"],
                "model": model,
                "params": params,
                "history": history,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "epochs_run": row["epochs_run"],
                "selection_metric": selection_metric,
            }

    df_results = pd.DataFrame(rows).sort_values(
        selection_metric,
        ascending=not maximize,
    ).reset_index(drop=True)
    return df_results, best


def run_baseline_grid_search(
    dfs: dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
    stations: Sequence[str] | None = None,
    grid: dict | None = None,
    epochs: int = 100,
    batch_size: int = 4,
    patience: int = 15,
    smooth_aod: bool = False,
    smooth_window: int = 7,
    inverse_scale: bool = True,
    selection_metric: str = "val_R2",
    results_prefix: str = "03_grid",
    verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan baseline grid search lintas stasiun dan simpan semua output."""
    stations = list(stations or C.HEALTHY_STATIONS)
    per_station_results: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for station in stations:
        data = build_pipeline(
            dfs[station],
            feature_cols,
            smooth_aod=smooth_aod,
            smooth_window=smooth_window,
        )
        df_res, best = grid_search(
            data,
            grid=grid,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            inverse_scale=inverse_scale,
            selection_metric=selection_metric,
            verbose=verbose,
        )

        df_res.insert(0, "station", station)
        metrics_path = C.METRICS_DIR / f"{results_prefix}_{station}.csv"
        model_path = C.MODEL_DIR / f"{results_prefix}_{station}.keras"

        df_res.to_csv(metrics_path, index=False)
        best["model"].save(model_path)

        per_station_results.append(df_res)
        summary_rows.append({
            "station": station,
            **best["params"],
            **{f"val_{k}": v for k, v in best["val_metrics"].items()},
            **{f"test_{k}": v for k, v in best["test_metrics"].items()},
            "epochs_run": best["epochs_run"],
            "selection_metric": best["selection_metric"],
            "selection_score": best["score"],
            "metrics_path": str(metrics_path),
            "model_path": str(model_path),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(
        "test_R2",
        ascending=False,
    ).reset_index(drop=True)
    df_all = pd.concat(per_station_results, ignore_index=True)

    df_summary.to_csv(C.METRICS_DIR / f"{results_prefix}_best_summary.csv", index=False)
    df_all.to_csv(C.METRICS_DIR / f"{results_prefix}_all_results.csv", index=False)
    return df_summary, df_all
