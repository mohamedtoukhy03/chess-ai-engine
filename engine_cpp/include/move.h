#pragma once

#include "types.h"
#include <string>

// ============================================================
// Move: 16-bit compact encoding
//
// bits  0-5:   from square
// bits  6-11:  to square
// bits 12-13:  promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
// bits 14-15:  flags (0=normal, 1=promotion, 2=en passant, 3=castling)
// ============================================================

enum MoveFlag : uint16_t {
    MOVE_NORMAL     = 0 << 14,
    MOVE_PROMOTION  = 1 << 14,
    MOVE_EN_PASSANT = 2 << 14,
    MOVE_CASTLING   = 3 << 14,
};

struct Move {
    uint16_t data;

    constexpr Move() : data(0) {}
    constexpr explicit Move(uint16_t d) : data(d) {}

    // --- Factory methods ---

    static Move make(Square from, Square to) {
        return Move(uint16_t(from) | (uint16_t(to) << 6));
    }

    static Move make_promotion(Square from, Square to, PieceType promo) {
        // Maps KNIGHT=1→0, BISHOP=2→1, ROOK=3→2, QUEEN=4→3
        return Move(uint16_t(from) | (uint16_t(to) << 6) |
                    (uint16_t(promo - KNIGHT) << 12) | MOVE_PROMOTION);
    }

    static Move make_en_passant(Square from, Square to) {
        return Move(uint16_t(from) | (uint16_t(to) << 6) | MOVE_EN_PASSANT);
    }

    static Move make_castling(Square king_from, Square king_to) {
        return Move(uint16_t(king_from) | (uint16_t(king_to) << 6) | MOVE_CASTLING);
    }

    // --- Accessors ---

    Square   from()  const { return Square(data & 0x3F); }
    Square   to()    const { return Square((data >> 6) & 0x3F); }
    uint16_t flags() const { return data & 0xC000; }

    PieceType promotion_type() const {
        return PieceType(((data >> 12) & 3) + KNIGHT);
    }

    bool is_promotion()   const { return flags() == MOVE_PROMOTION; }
    bool is_en_passant()  const { return flags() == MOVE_EN_PASSANT; }
    bool is_castling()    const { return flags() == MOVE_CASTLING; }

    bool operator==(Move other) const { return data == other.data; }
    bool operator!=(Move other) const { return data != other.data; }
    explicit operator bool() const { return data != 0; }

    // UCI-format string: "e2e4", "e7e8q"
    std::string to_string() const {
        std::string s = square_to_string(from()) + square_to_string(to());
        if (is_promotion()) {
            const char promo_chars[] = "nbrq";
            s += promo_chars[(data >> 12) & 3];
        }
        return s;
    }
};

constexpr Move MOVE_NONE = Move();

// ============================================================
// MoveList: stack-allocated move buffer (max 256)
// ============================================================

struct MoveList {
    Move moves[256];
    int count = 0;

    void add(Move m) {
        assert(count < 256);
        moves[count++] = m;
    }

    Move& operator[](int i)             { return moves[i]; }
    const Move& operator[](int i) const { return moves[i]; }

    Move*       begin()       { return moves; }
    Move*       end()         { return moves + count; }
    const Move* begin() const { return moves; }
    const Move* end()   const { return moves + count; }
};
