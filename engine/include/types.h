#pragma once

#include <cstdint>
#include <string>
#include <cassert>

// ============================================================
// Fundamental types for the chess engine
// ============================================================

using Bitboard = uint64_t;

// --- Color ---

enum Color : int {
    WHITE = 0,
    BLACK = 1,
    COLOR_NB = 2
};

constexpr Color operator~(Color c) { return Color(c ^ 1); }

// --- Piece types & pieces ---

enum PieceType : int {
    PAWN   = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK   = 3,
    QUEEN  = 4,
    KING   = 5,
    PIECE_TYPE_NB = 6,
    NO_PIECE_TYPE = 6
};

enum Piece : int {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 6, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE = 12,
    PIECE_NB = 12
};

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece(c * 6 + pt);
}

constexpr Color piece_color(Piece p) {
    assert(p != NO_PIECE);
    return Color(p / 6);
}

constexpr PieceType piece_type(Piece p) {
    assert(p != NO_PIECE);
    return PieceType(p % 6);
}

// --- Squares (LERF: Little-Endian Rank-File, A1=0, H8=63) ---

enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQUARE = 64,
    SQUARE_NB = 64
};

constexpr int square_rank(Square s) { return s >> 3; }
constexpr int square_file(Square s) { return s & 7; }
constexpr Square make_square(int file, int rank) { return Square(rank * 8 + file); }

constexpr Square operator+(Square s, int d) { return Square(int(s) + d); }
constexpr Square operator-(Square s, int d) { return Square(int(s) - d); }
inline Square& operator++(Square& s) { return s = Square(int(s) + 1); }

// --- Castling rights (4-bit bitmask) ---

enum CastlingRights : int {
    NO_CASTLING    = 0,
    WHITE_OO       = 1,  // White kingside
    WHITE_OOO      = 2,  // White queenside
    BLACK_OO       = 4,  // Black kingside
    BLACK_OOO      = 8,  // Black queenside
    ALL_CASTLING   = 15,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
};

constexpr CastlingRights operator|(CastlingRights a, CastlingRights b) {
    return CastlingRights(int(a) | int(b));
}
constexpr CastlingRights operator&(CastlingRights a, CastlingRights b) {
    return CastlingRights(int(a) & int(b));
}
constexpr CastlingRights operator~(CastlingRights a) {
    return CastlingRights(~int(a) & 0xF);
}
inline CastlingRights& operator|=(CastlingRights& a, CastlingRights b) {
    return a = a | b;
}
inline CastlingRights& operator&=(CastlingRights& a, CastlingRights b) {
    return a = a & b;
}

// --- Directions ---

enum Direction : int {
    NORTH =  8,
    SOUTH = -8,
    EAST  =  1,
    WEST  = -1,
    NORTH_EAST =  9,
    NORTH_WEST =  7,
    SOUTH_EAST = -7,
    SOUTH_WEST = -9
};

// --- Piece characters for FEN / display ---

constexpr char PIECE_CHAR[] = "PNBRQKpnbrqk ";

inline char piece_to_char(Piece p) { return PIECE_CHAR[p]; }

inline Piece char_to_piece(char c) {
    for (int i = 0; i < PIECE_NB; ++i)
        if (PIECE_CHAR[i] == c) return Piece(i);
    return NO_PIECE;
}

// --- Square string conversion ---

inline char file_to_char(int f) { return 'a' + f; }
inline char rank_to_char(int r) { return '1' + r; }

inline std::string square_to_string(Square s) {
    if (s == NO_SQUARE) return "-";
    return std::string(1, file_to_char(square_file(s))) +
           std::string(1, rank_to_char(square_rank(s)));
}

inline Square string_to_square(const std::string& s) {
    if (s == "-" || s.size() < 2) return NO_SQUARE;
    int file = s[0] - 'a';
    int rank = s[1] - '1';
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return NO_SQUARE;
    return make_square(file, rank);
}
