# Chess AI - Neural Network Training
import os
from pathlib import Path

# Project root (chess-engine/)
PROJECT_ROOT = Path(__file__).parent.parent

## Configuration
BOARD_CHANNELS = 12      # 6 piece types × 2 colors
BOARD_SIZE = 8           # 8×8 board
INPUT_CHANNELS = 12      # One-hot planes per piece type/color

## CNN Architecture
NUM_RESIDUAL_BLOCKS = 6
NUM_FILTERS = 128
FC_HIDDEN = 256

## Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
TRAIN_SPLIT = 0.9

## Paths (relative to project root)
PGN_DIR = str(PROJECT_ROOT / "data" / "pgn")
DATASET_PATH = str(PROJECT_ROOT / "data" / "dataset.pt")
MODEL_SAVE_PATH = str(PROJECT_ROOT / "models" / "best_model.pth")
ONNX_EXPORT_PATH = str(PROJECT_ROOT / "models" / "chess_eval.onnx")

## Stockfish (for generating evaluation labels)
STOCKFISH_PATH = "/usr/bin/stockfish"  # Adjust to your system
STOCKFISH_DEPTH = 12
NUM_POSITIONS = 400000    # Number of positions to generate
