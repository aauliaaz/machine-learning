"""Arsitektur LSTM dan helper training."""
from __future__ import annotations

from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

from . import config as C


def build_lstm(
    input_shape: tuple[int, int],
    lstm_units: int = 32,
    dropout_rate: float = 0.0,
    optimizer: Literal["adam", "rmsprop"] = "adam",
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
) -> Sequential:
    """Arsitektur sama dengan skripsi (Gambar 5) — 2 LSTM + Dense(25) + Dense(1).

    `clipnorm` ditambahkan untuk mencegah exploding gradient (NaN) saat
    dikombinasikan dengan aktivasi relu pada LSTM.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, activation="relu", return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation="relu", return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    opt_kwargs = {"learning_rate": learning_rate, "clipnorm": clipnorm}
    opt = Adam(**opt_kwargs) if optimizer == "adam" else RMSprop(**opt_kwargs)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 4,
    patience: int = 15,
    checkpoint_path: str | None = None,
    verbose: int = 0,
):
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,  # time-series: jangan shuffle
    )
    return history


def set_seed(seed: int = C.RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
