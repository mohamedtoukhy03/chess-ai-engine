"""Train TensorFlow SE-ResNet model from chessData.csv."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import mixed_precision

from config import (
    CHECKPOINT_PATH,
    CSV_PATH,
    DATASET_CACHE_PATH,
    EPOCHS,
    LEARNING_RATE,
    N_ROWS,
    RANDOM_SEED,
    USE_MIXED_PRECISION,
    VAL_SPLIT,
)
from dataset import (
    load_dataset_cache,
    make_dataset_from_arrays,
    preprocess_csv_to_numpy,
    save_dataset_cache,
)
from model import build_model


def plot_training_curves(history: keras.callbacks.History, output_dir: str) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history_dict.get("loss", []), label="Train Loss")
    axes[0].plot(epochs, history_dict.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss (Huber)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history_dict.get("mae", []), label="Train MAE")
    axes[1].plot(epochs, history_dict.get("val_mae", []), label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_validation_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: str
) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    residuals = y_pred - y_true

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true)))) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    sign_acc = float(np.mean((y_true > 0) == (y_pred > 0)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_true, y_pred, s=8, alpha=0.25)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1)
    axes[0].set_title("Predicted vs True (Validation)")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")
    axes[0].grid(alpha=0.3)
    axes[0].text(
        0.02,
        0.98,
        f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}\nSignAcc={sign_acc:.4f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    axes[1].hist(residuals, bins=60, alpha=0.85)
    axes[1].set_title("Residual Distribution (Pred - True)")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "validation_diagnostics.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def train() -> None:
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16")

    if os.path.isfile(DATASET_CACHE_PATH):
        print(f"Loading cached dataset: {DATASET_CACHE_PATH}")
        boards, meta, y = load_dataset_cache()
    else:
        if not os.path.isfile(CSV_PATH):
            raise FileNotFoundError(
                f"Missing dataset CSV at {CSV_PATH}. Place chessData.csv under data/."
            )
        boards, meta, y = preprocess_csv_to_numpy(CSV_PATH, N_ROWS)
        save_dataset_cache(boards, meta, y)

    n = len(y)
    val_n = max(1, int(VAL_SPLIT * n))
    train_n = n - val_n

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    tr_idx, va_idx = perm[:train_n], perm[train_n:]

    train_ds = make_dataset_from_arrays(
        boards[tr_idx], meta[tr_idx], y[tr_idx], shuffle=True
    )
    val_ds = make_dataset_from_arrays(
        boards[va_idx], meta[va_idx], y[va_idx], shuffle=False
    )

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH, save_best_only=True, monitor="val_loss", mode="min"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    model.summary()
    print(f"Train: {train_n:,} | Val: {val_n:,}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    output_dir = os.path.dirname(CHECKPOINT_PATH)
    plot_training_curves(history, output_dir)

    print("Running validation diagnostics ...")
    y_pred = model.predict([boards[va_idx], meta[va_idx]], verbose=0).reshape(-1)
    y_true = y[va_idx].reshape(-1)
    plot_validation_diagnostics(y_true, y_pred, output_dir)

    print(f"Done. Best checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
