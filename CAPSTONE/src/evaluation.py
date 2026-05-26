from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   r2_score(y_true, y_pred),
        "MSE":  mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  mean_absolute_error(y_true, y_pred),
    }


def plot_loss(history, save_path: Path | str | None = None) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train MSE")
    plt.plot(history.history["val_loss"], label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled)")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def plot_pred_vs_actual(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual PM2.5",
    save_path: Path | str | None = None,
) -> None:
    plt.figure(figsize=(13, 4))
    plt.plot(dates, y_true, label="Actual", alpha=0.6)
    plt.plot(dates, y_pred, label="Predicted", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
