"""
Dataset generation for chess CNN training.

Two modes:
  1. PGN parsing: Extract positions from Grandmaster PGN files
  2. Stockfish evaluation: Generate position labels using Stockfish engine

The board is encoded as a (18, 8, 8) float tensor:
  Planes 0-5:   White P, N, B, R, Q, K
  Planes 6-11:  Black P, N, B, R, Q, K
  Plane 12:     Side to move (all ones if White to move, else zeros)
  Plane 13:     White can castle kingside
  Plane 14:     White can castle queenside
  Plane 15:     Black can castle kingside
  Plane 16:     Black can castle queenside
  Plane 17:     En-passant target square (single 1 on target, if available)

Evaluations are normalized to [-1, 1] via a WDL-style expected score curve.
"""

import os
import sys
import math
import random
import atexit
import multiprocessing as mp
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
    Convert a python-chess Board to a (18, 8, 8) float tensor.

    White pieces → planes 0-5
    Black pieces → planes 6-11
    Context state  → planes 12-17
    """
    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = square // 8
            file = square % 8
            plane = PIECE_TO_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            tensor[plane, rank, file] = 1.0

    # Plane 12: side to move (white=1, black=0)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Planes 13-16: castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # Plane 17: en-passant target square
    if board.ep_square is not None:
        rank = board.ep_square // 8
        file = board.ep_square % 8
        tensor[17, rank, file] = 1.0

    return tensor


def cp_to_eval(cp: int) -> float:
    """
    Convert centipawn score to [-1, 1] via WDL-style expectation.

    Maps cp -> expected win probability with a logistic curve, then shifts
    to symmetric [-1, 1] to match the network's tanh output range.
    """
    win_prob = 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))
    return (2.0 * win_prob) - 1.0


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

    Uses multi-process parallelism so each worker owns one Stockfish process.
    """
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        return []

    candidate_fens = collect_candidate_fens(pgn_dir, num_positions)
    if not candidate_fens:
        return []

    num_workers = max(1, (os.cpu_count() or 1))
    print(f"Evaluating {len(candidate_fens)} positions with {num_workers} Stockfish workers (depth={depth})...")

    positions = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_stockfish_worker,
        initargs=(stockfish_path, depth),
    ) as pool:
        pbar = tqdm(total=min(num_positions, len(candidate_fens)), desc="Stockfish evals")
        for result in pool.imap_unordered(_evaluate_fen_worker, candidate_fens, chunksize=32):
            if result is None:
                continue
            fen, label = result
            tensor = board_to_tensor(chess.Board(fen))
            positions.append((tensor, label))
            pbar.update(1)
            if len(positions) >= num_positions:
                break
        pbar.close()

    print(f"Generated {len(positions)} Stockfish-evaluated positions")
    return positions


def collect_candidate_fens(pgn_dir: str, target_count: int) -> list:
    """Collect candidate FENs from PGNs before parallel Stockfish evaluation."""
    fens = []
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith(".pgn")]
    open_files = [open(os.path.join(pgn_dir, f), encoding="utf-8", errors="replace") for f in pgn_files]
    print(f"Collecting candidate positions from {len(pgn_files)} PGN files...")
    try:
        while len(fens) < target_count and open_files:
            for f in list(open_files):
                game = chess.pgn.read_game(f)
                if game is None:
                    open_files.remove(f)
                    continue

                board = game.board()
                moves = list(game.mainline_moves())
                for i, move in enumerate(moves):
                    board.push(move)
                    # Skip unstable opening plies, then sample every 2 plies.
                    if i < 8 or i % 2 != 0:
                        continue
                    fens.append(board.fen())
                    if len(fens) >= target_count:
                        break
                if len(fens) >= target_count:
                    break
    finally:
        for f in open_files:
            if not f.closed:
                f.close()

    print(f"Collected {len(fens)} candidate positions")
    return fens


_WORKER_ENGINE = None
_WORKER_DEPTH = None


def _init_stockfish_worker(stockfish_path: str, depth: int):
    """Initialize one persistent Stockfish process per worker."""
    global _WORKER_ENGINE, _WORKER_DEPTH
    _WORKER_DEPTH = depth
    _WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def _cleanup():
        try:
            if _WORKER_ENGINE is not None:
                _WORKER_ENGINE.quit()
        except Exception:
            pass

    atexit.register(_cleanup)


def _evaluate_fen_worker(fen: str):
    """Evaluate one FEN in a worker process; returns (fen, label) or None."""
    try:
        board = chess.Board(fen)
        info = _WORKER_ENGINE.analyse(board, chess.engine.Limit(depth=_WORKER_DEPTH))
        score = info["score"].white()
        if score.is_mate():
            label = mate_to_eval(score.mate())
        else:
            label = cp_to_eval(score.score())
        return fen, label
    except Exception:
        return None


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
