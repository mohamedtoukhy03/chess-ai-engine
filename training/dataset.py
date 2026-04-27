"""Notebook-aligned CSV -> tensor preprocessing utilities."""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
from numba import njit, prange

from config import (
    BATCH_SIZE,
    CLIP_CP,
    COL_EVAL,
    COL_FEN,
    DATASET_CACHE_PATH,
    EVAL_SCALE_CP,
    IS_CHECK_WORKERS,
    META_DIM,
)

_PIECE_LUT = np.full(128, -1, dtype=np.int8)
for _char, _idx in zip("PNBRQK", range(0, 6)):
    _PIECE_LUT[ord(_char)] = _idx
for _char, _idx in zip("pnbrqk", range(6, 12)):
    _PIECE_LUT[ord(_char)] = _idx


def evaluations_to_centipawns(eval_series: pd.Series) -> np.ndarray:
    series = eval_series.astype(str).str.strip()
    out = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    mate_m = series.str.extract(r"^#\s*([+-]?\d+)\s*$", expand=False)
    has_mate = mate_m.notna()
    if has_mate.any():
        mate_vals = mate_m[has_mate].astype(np.int32).to_numpy()
        idx = has_mate.to_numpy()
        mate_vals_f = mate_vals.astype(np.float64)
        out[idx] = np.sign(mate_vals_f) * (10_000.0 + np.abs(mate_vals_f))
    return out


def centipawns_to_target(cp: np.ndarray) -> np.ndarray:
    cp = np.clip(cp.astype(np.float64), -CLIP_CP, CLIP_CP)
    win_prob = 1.0 / (1.0 + np.power(10.0, -cp / EVAL_SCALE_CP))
    return (2.0 * win_prob - 1.0).astype(np.float32)


@njit(parallel=True, cache=True)
def _encode_board_meta_batch(
    fen_bytes: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    piece_lut: np.ndarray,
    out_board: np.ndarray,
    out_meta: np.ndarray,
):
    # Encode to the same canonical perspective used by engine_cpp:
    # side-to-move is always represented as "white" after canonicalization.
    n = lengths.shape[0]
    for i in prange(n):
        start = offsets[i]
        length = lengths[i]
        spaces = np.empty(5, dtype=np.int32)
        sp_count = 0
        for j in range(length):
            c = fen_bytes[start + j]
            if c == 32:
                spaces[sp_count] = j
                sp_count += 1
                if sp_count == 5:
                    break
        if sp_count < 5:
            continue

        board_s0 = 0
        board_s1 = spaces[0]
        turn_s0 = spaces[0] + 1
        turn_s1 = spaces[1]
        cast_s0 = spaces[1] + 1
        cast_s1 = spaces[2]
        ep_s0 = spaces[2] + 1
        ep_s1 = spaces[3]

        is_black_to_move = turn_s1 > turn_s0 and fen_bytes[start + turn_s0] == 98

        rank = 0
        file = 0
        for j in range(board_s0, board_s1):
            ch = fen_bytes[start + j]
            if ch == 47:
                rank += 1
                file = 0
                continue
            if rank >= 8:
                break
            if 49 <= ch <= 56:
                empties = ch - 48
                for _ in range(empties):
                    if file >= 8:
                        break
                    for p in range(12):
                        out_board[i, rank, file, p] = 0.0
                    file += 1
            else:
                if file < 8:
                    pidx = -1
                    if ch < 128:
                        pidx = piece_lut[ch]
                    t_rank = rank
                    t_file = file
                    t_pidx = pidx

                    if is_black_to_move:
                        # Rotate board and swap piece colors in channel space.
                        t_rank = 7 - t_rank
                        t_file = 7 - t_file
                        if 0 <= t_pidx < 6:
                            t_pidx = t_pidx + 6
                        elif 6 <= t_pidx < 12:
                            t_pidx = t_pidx - 6

                    for p in range(12):
                        out_board[i, t_rank, t_file, p] = 0.0
                    if 0 <= t_pidx < 12:
                        out_board[i, t_rank, t_file, t_pidx] = 1.0
                    file += 1

        # Canonical side-to-move perspective: always "white to move".
        out_meta[i, 0] = 1.0

        cw_k = cw_q = cb_k = cb_q = 0.0
        for j in range(cast_s0, cast_s1):
            ch = fen_bytes[start + j]
            if ch == 75:
                cw_k = 1.0
            elif ch == 81:
                cw_q = 1.0
            elif ch == 107:
                cb_k = 1.0
            elif ch == 113:
                cb_q = 1.0
        if not is_black_to_move:
            out_meta[i, 1] = cw_k
            out_meta[i, 2] = cw_q
            out_meta[i, 3] = cb_k
            out_meta[i, 4] = cb_q
        else:
            # Swap castling rights ordering to stay canonical.
            out_meta[i, 1] = cb_k
            out_meta[i, 2] = cb_q
            out_meta[i, 3] = cw_k
            out_meta[i, 4] = cw_q

        ep_pres = 0.0
        ep_file_norm = 0.0
        if ep_s1 > ep_s0:
            ech = fen_bytes[start + ep_s0]
            if ech != 45:
                ep_pres = 1.0
                if 97 <= ech <= 104:
                    ep_file = ech - 97
                    if is_black_to_move:
                        ep_file = 7 - ep_file
                    ep_file_norm = ep_file / 7.0
        out_meta[i, 5] = ep_pres
        out_meta[i, 6] = ep_file_norm
        out_meta[i, 7] = 0.0


def pack_fens_for_numba(fens: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(fens)
    lengths = np.empty(n, dtype=np.int32)
    for i, fen in enumerate(fens):
        lengths[i] = len(fen)

    total = int(lengths.sum())
    buf = np.empty(total, dtype=np.uint8)
    offsets = np.empty(n, dtype=np.int32)
    off = 0
    for i, fen in enumerate(fens):
        offsets[i] = off
        for j, ch in enumerate(fen):
            buf[off + j] = ord(ch)
        off += lengths[i]
    return buf, offsets, lengths


def fens_to_tensors_batch(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = len(fens)
    boards = np.zeros((n, 8, 8, 12), dtype=np.float32)
    meta = np.zeros((n, META_DIM), dtype=np.float32)
    buf, offsets, lengths = pack_fens_for_numba(fens)
    _encode_board_meta_batch(buf, offsets, lengths, _PIECE_LUT, boards, meta)
    return boards, meta


def _is_check_one_fen(fen: str) -> float:
    try:
        import chess

        return 1.0 if chess.Board(fen).is_check() else 0.0
    except Exception:
        return 0.0


def fill_in_check_metadata(fens: list[str], meta: np.ndarray, workers: int) -> None:
    if workers <= 0 or meta.shape[1] < 8:
        return
    try:
        import chess  # noqa: F401
    except ImportError:
        return

    chunksize = max(1, len(fens) // (workers * 64))
    with Pool(workers) as pool:
        meta[:, 7] = np.asarray(
            pool.map(_is_check_one_fen, fens, chunksize=chunksize), dtype=np.float32
        )


def preprocess_csv_to_numpy(csv_path: str, n_rows: int):
    print(f"Reading {n_rows:,} rows from {csv_path} ...")
    df = pd.read_csv(csv_path, nrows=n_rows, usecols=[COL_FEN, COL_EVAL])
    df = df.dropna(subset=[COL_FEN, COL_EVAL])
    fens = df[COL_FEN].tolist()
    cp = evaluations_to_centipawns(df[COL_EVAL])
    y = centipawns_to_target(cp)

    # Make labels side-to-move relative to match canonical encoding.
    stm_black = np.array(
        [1.0 if fen.split(" ")[1] == "b" else 0.0 for fen in fens], dtype=np.float32
    )
    y = y * (1.0 - 2.0 * stm_black)
    del df

    print(f"Parsing {len(fens):,} FENs with Numba ...")
    boards, meta = fens_to_tensors_batch(fens)
    if IS_CHECK_WORKERS > 0:
        print("Computing in-check metadata (python-chess) ...")
        fill_in_check_metadata(fens, meta, IS_CHECK_WORKERS)
    return boards, meta, y


def save_dataset_cache(boards: np.ndarray, meta: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(DATASET_CACHE_PATH, boards=boards, meta=meta, y=y)
    print(f"Saved dataset cache: {DATASET_CACHE_PATH}")


def load_dataset_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(DATASET_CACHE_PATH)
    return data["boards"], data["meta"], data["y"]


def make_dataset_from_arrays(
    boards: np.ndarray, meta: np.ndarray, y: np.ndarray, shuffle: bool
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(((boards, meta), y))
    if shuffle:
        ds = ds.shuffle(min(100_000, len(y)), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)
