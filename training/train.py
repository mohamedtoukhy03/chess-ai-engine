"""
Training script for the Chess Evaluation CNN.

Features:
  - Supervised learning on position/evaluation pairs
  - Train/validation split with early stopping
  - Metrics: MSE loss, MAE, Sign Accuracy
  - Learning rate scheduling
  - Best model checkpointing
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    TRAIN_SPLIT, MODEL_SAVE_PATH, DATASET_PATH
)
from model import ChessEvalNet, count_parameters
from dataset import ChessDataset, load_dataset, build_dataset


def train():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Load dataset ---
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        tensors, labels = load_dataset()
    else:
        print("No dataset found. Building dataset...")
        tensors, labels = build_dataset()

    print(f"Dataset: {len(labels)} positions")
    print(f"Label distribution: mean={labels.mean():.3f}, std={labels.std():.3f}\n")

    # --- Train/Val split ---
    dataset = ChessDataset(tensors, labels)
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {train_size} | Val: {val_size}\n")

    # --- Model ---
    model = ChessEvalNet().to(device)
    print(f"Model: {count_parameters(model):,} parameters\n")

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.MSELoss()

    # --- Training history ---
    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
        "val_sign_acc": []
    }

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- Training loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}")
        for boards, evals in pbar:
            boards, evals = boards.to(device), evals.to(device)

            optimizer.zero_grad()
            preds = model(boards)
            loss = criterion(preds, evals)
            loss.backward()
            optimizer.step()

            batch_size = boards.size(0)
            train_loss_sum += loss.item() * batch_size
            train_mae_sum += (preds - evals).abs().sum().item()
            train_count += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_count
        train_mae = train_mae_sum / train_count

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_sign_correct = 0
        val_sign_total = 0
        val_count = 0

        with torch.no_grad():
            for boards, evals in val_loader:
                boards, evals = boards.to(device), evals.to(device)
                preds = model(boards)
                loss = criterion(preds, evals)

                batch_size = boards.size(0)
                val_loss_sum += loss.item() * batch_size
                val_mae_sum += (preds - evals).abs().sum().item()
                val_count += batch_size

                # Sign accuracy (ignore near-zero evaluations)
                mask = evals.abs() > 0.05
                if mask.sum() > 0:
                    val_sign_correct += ((preds[mask] > 0) == (evals[mask] > 0)).sum().item()
                    val_sign_total += mask.sum().item()

        val_loss = val_loss_sum / val_count
        val_mae = val_mae_sum / val_count
        sign_acc = val_sign_correct / max(val_sign_total, 1)

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["val_sign_acc"].append(sign_acc)

        print(f"  Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | "
              f"Sign Acc: {sign_acc:.1%}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        print()

    # --- Plot training curves ---
    plot_history(history)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


def plot_history(history: dict):
    """Plot training/validation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history["train_mae"], label="Train")
    axes[1].plot(history["val_mae"], label="Val")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Sign Accuracy
    axes[2].plot(history["val_sign_acc"], label="Val", color="green")
    axes[2].set_title("Sign Accuracy (Val)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.4, 1.0)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../models/training_curves.png", dpi=150)
    print("\nTraining curves saved to ../models/training_curves.png")
    plt.close()


if __name__ == "__main__":
    train()
