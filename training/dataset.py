"""
Dataset generation for chess CNN training.

Two modes:
  1. PGN parsing: Extract positions from Grandmaster PGN files
  2. Stockfish evaluation: Generate position labels using Stockfish engine

The board is encoded as a (12, 8, 8) float tensor:
  Planes 0-5:   White P, N, B, R, Q, K
  Planes 6-11:  Black P, N, B, R, Q, K

Evaluations are normalized to [-1, 1] via tanh(centipawn / 400).
"""

import os
import sys
import math
import random
import torch
import chess
import chess.pgn
import chess.engine
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    PGN_DIR, DATASET_PATH, STOCKFISH_PATH,
    STOCKFISH_DEPTH, NUM_POSITIONS, BATCH_SIZE
)


# ============================================================
# Board encoding
# ============================================================

PIECE_TO_PLANE = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board to a (12, 8, 8) float tensor.

    White pieces → planes 0-5
    Black pieces → planes 6-11
    """
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = square // 8
            file = square % 8
            plane = PIECE_TO_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            tensor[plane, rank, file] = 1.0

    return tensor


def cp_to_eval(cp: int) -> float:
    """Convert centipawn score to [-1, 1] via tanh normalization."""
    return math.tanh(cp / 400.0)


def mate_to_eval(mate_in: int) -> float:
    """Convert mate score: positive = White winning, negative = Black winning."""
    if mate_in > 0:
        return 1.0
    elif mate_in < 0:
        return -1.0
    return 0.0


# ============================================================
# Dataset generation from PGN files
# ============================================================

def extract_positions_from_pgn(pgn_dir: str, max_positions: int) -> list:
    """
    Parse PGN files and extract board positions with game results.

    Labels from game results: 1-0 → +1.0, 0-1 → -1.0, 1/2-1/2 → 0.0
    """
    positions = []
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        return positions

    print(f"Parsing {len(pgn_files)} PGN files simultaneously...")
    open_files = [open(os.path.join(pgn_dir, f), encoding='utf-8', errors='replace') for f in pgn_files]

    try:
        while len(positions) < max_positions and open_files:
            for f in list(open_files):
                game = chess.pgn.read_game(f)
                if game is None:
                    open_files.remove(f)
                    continue

                result = game.headers.get("Result", "*")
                if result == "1-0":
                    label = 1.0
                elif result == "0-1":
                    label = -1.0
                elif result == "1/2-1/2":
                    label = 0.0
                else:
                    continue

                board = game.board()
                moves = list(game.mainline_moves())

                # Sample positions from the game
                start = min(10, len(moves) // 4)
                end = max(start + 1, len(moves) - 5)
                sample_indices = sorted(random.sample(
                    range(start, end),
                    min(8, end - start)
                ))

                for i, move in enumerate(moves):
                    board.push(move)
                    if i in sample_indices:
                        tensor = board_to_tensor(board)
                        positions.append((tensor, label))

                if len(positions) >= max_positions:
                    break
    finally:
        for f in open_files:
            if not f.closed:
                f.close()

    print(f"Extracted {len(positions)} positions from PGN files")
    return positions


# ============================================================
# Dataset generation using Stockfish evaluations
# ============================================================

def generate_stockfish_dataset(pgn_dir: str, num_positions: int,
                                stockfish_path: str, depth: int) -> list:
    """
    Parse PGN files and evaluate positions with Stockfish for accurate labels.
    """
    positions = []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"Could not start Stockfish at {stockfish_path}: {e}")
        print("Falling back to PGN result labels...")
        return extract_positions_from_pgn(pgn_dir, num_positions)

    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        return positions

    print(f"Processing {len(pgn_files)} files in round-robin with Stockfish (depth={depth})...")
    open_files = [open(os.path.join(pgn_dir, f), encoding='utf-8', errors='replace') for f in pgn_files]

    try:
        pbar = tqdm(total=num_positions, desc="Positions")
        while len(positions) < num_positions and open_files:
            for f in list(open_files):
                game = chess.pgn.read_game(f)
                if game is None:
                    open_files.remove(f)
                    continue

                board = game.board()
                moves = list(game.mainline_moves())

                for i, move in enumerate(moves):
                    board.push(move)

                    # Skip very early opening and use every 3rd position
                    if i < 8 or i % 3 != 0:
                        continue

                    # Evaluate with Stockfish
                    info = engine.analyse(board, chess.engine.Limit(depth=depth))
                    score = info["score"].white()

                    if score.is_mate():
                        label = mate_to_eval(score.mate())
                    else:
                        label = cp_to_eval(score.score())

                    tensor = board_to_tensor(board)
                    positions.append((tensor, label))
                    pbar.update(1)

                    if len(positions) >= num_positions:
                        break
                        
                if len(positions) >= num_positions:
                    break
        pbar.close()
    finally:
        for f in open_files:
            if not f.closed:
                f.close()
        engine.quit()

    print(f"Generated {len(positions)} Stockfish-evaluated positions")
    return positions


# ============================================================
# PyTorch Dataset
# ============================================================

class ChessDataset(Dataset):
    """PyTorch dataset wrapping position tensors and evaluation labels."""

    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


def build_dataset(use_stockfish: bool = True) -> tuple:
    """
    Build and save the training dataset.

    Returns:
        (tensors, labels) tuple of torch tensors
    """
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    os.makedirs(PGN_DIR, exist_ok=True)

    if use_stockfish and os.path.exists(STOCKFISH_PATH):
        data = generate_stockfish_dataset(
            PGN_DIR, NUM_POSITIONS, STOCKFISH_PATH, STOCKFISH_DEPTH
        )
    else:
        data = extract_positions_from_pgn(PGN_DIR, NUM_POSITIONS)

    if not data:
        print("No data generated. Please add PGN files to", PGN_DIR)
        print("Generating synthetic random data for testing...")
        data = generate_synthetic_data(1000)

    random.shuffle(data)
    tensors = torch.stack([d[0] for d in data])
    labels = torch.tensor([d[1] for d in data], dtype=torch.float32).unsqueeze(1)

    # Save dataset
    torch.save({"tensors": tensors, "labels": labels}, DATASET_PATH)
    print(f"Dataset saved to {DATASET_PATH}")
    print(f"  Tensors shape: {tensors.shape}")
    print(f"  Labels shape:  {labels.shape}")
    print(f"  Label range:   [{labels.min():.3f}, {labels.max():.3f}]")

    return tensors, labels


def generate_synthetic_data(n: int) -> list:
    """Generate random board positions for testing the pipeline."""
    data = []
    for _ in range(n):
        board = chess.Board()
        # Play random moves
        num_moves = random.randint(5, 40)
        for _ in range(num_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(random.choice(legal))

        tensor = board_to_tensor(board)
        # Random evaluation (just for pipeline testing)
        label = random.uniform(-1.0, 1.0)
        data.append((tensor, label))

    return data


def load_dataset() -> tuple:
    """Load a previously saved dataset."""
    data = torch.load(DATASET_PATH, weights_only=True)
    return data["tensors"], data["labels"]


if __name__ == "__main__":
    print("Chess Dataset Generator")
    print("=======================\n")

    if "--synthetic" in sys.argv:
        print("Generating synthetic test dataset...")
        data = generate_synthetic_data(5000)
        random.shuffle(data)
        tensors = torch.stack([d[0] for d in data])
        labels = torch.tensor([d[1] for d in data], dtype=torch.float32).unsqueeze(1)
        torch.save({"tensors": tensors, "labels": labels}, DATASET_PATH)
        print(f"Saved {len(data)} synthetic positions to {DATASET_PATH}")
    else:
        build_dataset(use_stockfish=True)
