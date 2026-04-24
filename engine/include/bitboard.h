#pragma once

#include "types.h"

// ============================================================
// Bitboard utilities, precomputed attack tables, and Zobrist keys
// ============================================================

namespace BB {

// --- Bit manipulation (compiler intrinsics) ---

inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

inline Square lsb(Bitboard b) {
    assert(b);
    return Square(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
    assert(b);
    return Square(63 ^ __builtin_clzll(b));
}

inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

// --- Constants ---

constexpr Bitboard EMPTY  = 0ULL;
constexpr Bitboard ALL_SQ = ~0ULL;

constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;

constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

constexpr Bitboard NOT_FILE_A  = ~FILE_A_BB;
constexpr Bitboard NOT_FILE_H  = ~FILE_H_BB;
constexpr Bitboard NOT_FILE_AB = ~(FILE_A_BB | FILE_B_BB);
constexpr Bitboard NOT_FILE_GH = ~(FILE_G_BB | FILE_H_BB);

constexpr Bitboard FileBB[8] = {
    FILE_A_BB, FILE_B_BB, FILE_C_BB, FILE_D_BB,
    FILE_E_BB, FILE_F_BB, FILE_G_BB, FILE_H_BB
};

constexpr Bitboard RankBB[8] = {
    RANK_1_BB, RANK_2_BB, RANK_3_BB, RANK_4_BB,
    RANK_5_BB, RANK_6_BB, RANK_7_BB, RANK_8_BB
};

inline constexpr Bitboard square_bb(Square s) {
    return 1ULL << s;
}

// --- Safe shift operations (no wrapping around files) ---

inline constexpr Bitboard shift_north(Bitboard b) { return b << 8; }
inline constexpr Bitboard shift_south(Bitboard b) { return b >> 8; }
inline constexpr Bitboard shift_east(Bitboard b)  { return (b << 1) & NOT_FILE_A; }
inline constexpr Bitboard shift_west(Bitboard b)  { return (b >> 1) & NOT_FILE_H; }
inline constexpr Bitboard shift_ne(Bitboard b)    { return (b << 9) & NOT_FILE_A; }
inline constexpr Bitboard shift_nw(Bitboard b)    { return (b << 7) & NOT_FILE_H; }
inline constexpr Bitboard shift_se(Bitboard b)    { return (b >> 7) & NOT_FILE_A; }
inline constexpr Bitboard shift_sw(Bitboard b)    { return (b >> 9) & NOT_FILE_H; }

// --- Precomputed attack tables ---

extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];

// --- Sliding piece attacks (classical ray iteration) ---

Bitboard bishop_attacks(Square sq, Bitboard occupied);
Bitboard rook_attacks(Square sq, Bitboard occupied);
Bitboard queen_attacks(Square sq, Bitboard occupied);

// --- Line & Between bitboards (for pin/check geometry) ---

extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];

// --- Zobrist hashing ---

namespace Zobrist {
    extern uint64_t PieceSquare[PIECE_NB][SQUARE_NB];
    extern uint64_t EnPassant[8];
    extern uint64_t Castling[16];
    extern uint64_t SideToMove;
}

// --- Initialization (call once at startup) ---

void init();

// --- Debug: print bitboard as 8x8 grid ---

void print(Bitboard b);

} // namespace BB
