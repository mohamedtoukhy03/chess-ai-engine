#include "bitboard.h"

#include <iostream>
#include <iomanip>
#include <random>

namespace BB {

// ============================================================
// Global table storage
// ============================================================

Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];

namespace Zobrist {
    uint64_t PieceSquare[PIECE_NB][SQUARE_NB];
    uint64_t EnPassant[8];
    uint64_t Castling[16];
    uint64_t SideToMove;
}

// ============================================================
// Classical ray attack helpers
// ============================================================

namespace {

Bitboard ray_attack(Square sq, int df, int dr, Bitboard occupied) {
    Bitboard attacks = EMPTY;
    int f = square_file(sq) + df;
    int r = square_rank(sq) + dr;

    while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        Square s = make_square(f, r);
        Bitboard b = square_bb(s);
        attacks |= b;
        if (b & occupied) break;
        f += df;
        r += dr;
    }
    return attacks;
}

Bitboard compute_knight_attacks(Square sq) {
    Bitboard b = square_bb(sq);
    Bitboard attacks = EMPTY;
    attacks |= (b << 17) & NOT_FILE_A;
    attacks |= (b << 15) & NOT_FILE_H;
    attacks |= (b << 10) & NOT_FILE_AB;
    attacks |= (b <<  6) & NOT_FILE_GH;
    attacks |= (b >> 17) & NOT_FILE_H;
    attacks |= (b >> 15) & NOT_FILE_A;
    attacks |= (b >> 10) & NOT_FILE_GH;
    attacks |= (b >>  6) & NOT_FILE_AB;
    return attacks;
}

Bitboard compute_king_attacks(Square sq) {
    Bitboard b = square_bb(sq);
    return shift_north(b) | shift_south(b) |
           shift_east(b)  | shift_west(b)  |
           shift_ne(b)    | shift_nw(b)    |
           shift_se(b)    | shift_sw(b);
}

Bitboard compute_pawn_attacks(Square sq, Color c) {
    Bitboard b = square_bb(sq);
    if (c == WHITE)
        return shift_ne(b) | shift_nw(b);
    else
        return shift_se(b) | shift_sw(b);
}

} // anonymous namespace

// ============================================================
// Public sliding piece functions
// ============================================================

Bitboard bishop_attacks(Square sq, Bitboard occupied) {
    return ray_attack(sq,  1,  1, occupied) |  // NE
           ray_attack(sq,  1, -1, occupied) |  // SE
           ray_attack(sq, -1, -1, occupied) |  // SW
           ray_attack(sq, -1,  1, occupied);   // NW
}

Bitboard rook_attacks(Square sq, Bitboard occupied) {
    return ray_attack(sq,  0,  1, occupied) |  // N
           ray_attack(sq,  1,  0, occupied) |  // E
           ray_attack(sq,  0, -1, occupied) |  // S
           ray_attack(sq, -1,  0, occupied);   // W
}

Bitboard queen_attacks(Square sq, Bitboard occupied) {
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied);
}

// ============================================================
// Initialization
// ============================================================

void init() {
    // --- Piece attack tables ---
    for (int sq = 0; sq < SQUARE_NB; ++sq) {
        KnightAttacks[sq] = compute_knight_attacks(Square(sq));
        KingAttacks[sq]   = compute_king_attacks(Square(sq));
        PawnAttacks[WHITE][sq] = compute_pawn_attacks(Square(sq), WHITE);
        PawnAttacks[BLACK][sq] = compute_pawn_attacks(Square(sq), BLACK);
    }

    // --- Line & Between bitboards ---
    // Direction vectors for all 8 compass directions
    constexpr int dirs[8][2] = {
        { 0,  1}, { 1,  1}, { 1,  0}, { 1, -1},
        { 0, -1}, {-1, -1}, {-1,  0}, {-1,  1}
    };

    for (int s1 = 0; s1 < SQUARE_NB; ++s1) {
        for (int s2 = 0; s2 < SQUARE_NB; ++s2) {
            LineBB[s1][s2]    = EMPTY;
            BetweenBB[s1][s2] = EMPTY;
        }
    }

    for (int s1 = 0; s1 < SQUARE_NB; ++s1) {
        for (auto& dir : dirs) {
            int df = dir[0], dr = dir[1];
            Bitboard ray = EMPTY;

            int f = square_file(Square(s1)) + df;
            int r = square_rank(Square(s1)) + dr;

            while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
                Square s2 = make_square(f, r);
                ray |= square_bb(s2);

                // BetweenBB: squares strictly between s1 and s2
                Bitboard between = EMPTY;
                int bf = square_file(Square(s1)) + df;
                int br = square_rank(Square(s1)) + dr;
                while (make_square(bf, br) != s2) {
                    between |= square_bb(make_square(bf, br));
                    bf += df;
                    br += dr;
                }
                BetweenBB[s1][s2] = between;

                // LineBB: full line through both squares (reverse ray + s1 + forward ray)
                Bitboard rev = EMPTY;
                int rf = square_file(Square(s1)) - df;
                int rr = square_rank(Square(s1)) - dr;
                while (rf >= 0 && rf <= 7 && rr >= 0 && rr <= 7) {
                    rev |= square_bb(make_square(rf, rr));
                    rf -= df;
                    rr -= dr;
                }
                LineBB[s1][s2] = rev | square_bb(Square(s1)) | ray;

                f += df;
                r += dr;
            }
        }
    }

    // --- Zobrist hash keys (deterministic seed for reproducibility) ---
    std::mt19937_64 rng(0x1234567890ABCDEFULL);

    for (int p = 0; p < PIECE_NB; ++p)
        for (int sq = 0; sq < SQUARE_NB; ++sq)
            Zobrist::PieceSquare[p][sq] = rng();

    for (int f = 0; f < 8; ++f)
        Zobrist::EnPassant[f] = rng();

    for (int cr = 0; cr < 16; ++cr)
        Zobrist::Castling[cr] = rng();

    Zobrist::SideToMove = rng();
}

// ============================================================
// Debug: print bitboard as 8x8 grid
// ============================================================

void print(Bitboard b) {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r + 1) << " |";
        for (int f = 0; f <= 7; ++f) {
            Square sq = make_square(f, r);
            std::cout << (b & square_bb(sq) ? " X |" : "   |");
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n";
    std::cout << "  Hex: 0x" << std::hex << std::setfill('0')
              << std::setw(16) << b << std::dec << "\n\n";
}

} // namespace BB
