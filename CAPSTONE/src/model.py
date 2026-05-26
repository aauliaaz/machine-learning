from __future__ import annotations

from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
    activation: Literal["relu", "tanh"] = "relu",
    dense_units: int = 25,
) -> Sequential:

    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, activation=activation, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation=activation, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
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
    reduce_lr: bool = False,
    reduce_lr_patience: int = 7,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-5,
    verbose: int = 0,
):
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True))
    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=0,
            )
        )

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
