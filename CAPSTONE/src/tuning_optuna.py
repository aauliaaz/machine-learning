from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers import TPESampler

from . import config as C
from .evaluation import compute_metrics
from .model import build_lstm, set_seed, train_model
from .preprocessing import inverse_target


# Skor yang dikembalikan ketika trial menghasilkan NaN/Inf sangat rendah
# tapi bukan -inf agar TPE sampler tetap bisa learning arah ruang pencarian
_BAD_SCORE = -1e6


def _safe_predict(model, X: np.ndarray) -> np.ndarray | None:

    y_pred = model.predict(X, verbose=0).flatten()
    if not np.all(np.isfinite(y_pred)):
        return None
    return y_pred


def make_objective(
    data: dict,
    epochs: int = 100,
    batch_size: int = 4,
    patience: int = 15,
    inverse_scale: bool = True,
):
    n_features = data["X_train"].shape[2]

    def objective(trial: optuna.Trial) -> float:
        # Range learning_rate diperketat ke 1e-2 — relu+LSTM tidak stabil di lr tinggi.
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "optimizer":     trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
            "lstm_units":    trial.suggest_categorical("lstm_units", [16, 32, 64, 128]),
            "dropout_rate":  trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.05),
        }

        set_seed()
        model = build_lstm(input_shape=(C.LOOKBACK, n_features), **params)
        history = train_model(
            model,
            data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
            epochs=epochs, batch_size=batch_size, patience=patience,
            verbose=0,
        )

        # Cek apakah training divergen (loss NaN)
        if not np.all(np.isfinite(history.history.get("val_loss", [np.nan]))):
            return _BAD_SCORE

        y_pred = _safe_predict(model, data["X_val"])
        if y_pred is None:
            return _BAD_SCORE

        y_true = data["y_val"]
        if inverse_scale:
            y_pred = inverse_target(y_pred, data["scaler"], data["target_idx"], n_features)
            y_true = inverse_target(y_true, data["scaler"], data["target_idx"], n_features)

        if not (np.all(np.isfinite(y_pred)) and np.all(np.isfinite(y_true))):
            return _BAD_SCORE

        m = compute_metrics(y_true, y_pred)
        score = m["R2"]
        return score if np.isfinite(score) else _BAD_SCORE

    return objective


def run_optuna(
    data: dict,
    n_trials: int = 30,
    epochs: int = 100,
    batch_size: int = 4,
    patience: int = 15,
    seed: int = C.RANDOM_SEED,
) -> optuna.Study:
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(data, epochs, batch_size, patience),
        n_trials=n_trials,
      
        catch=(ValueError, RuntimeError),
    )
    return study


def fit_best(study: optuna.Study, data: dict, epochs: int = 100, batch_size: int = 4, patience: int = 15):

    n_features = data["X_train"].shape[2]
    set_seed()
    model = build_lstm(input_shape=(C.LOOKBACK, n_features), **study.best_params)
    history = train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        epochs=epochs, batch_size=batch_size, patience=patience,
        verbose=0,
    )
    return model, history
