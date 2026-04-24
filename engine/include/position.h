#pragma once

#include "types.h"
#include "bitboard.h"
#include "move.h"

#include <string>

// ============================================================
// UndoInfo: state saved before a move for unmake_move()
// ============================================================

struct UndoInfo {
    CastlingRights castling;
    Square         ep_square;
    int            halfmove;
    Piece          captured;   // The piece captured (or NO_PIECE)
    uint64_t       hash;
};

// ============================================================
// Position: complete board state
// ============================================================

class Position {
public:
    Position();

    // --- FEN ---
    void set_fen(const std::string& fen);
    std::string to_fen() const;
    void set_startpos();

    // --- Accessors ---
    Color     side_to_move()    const { return side_; }
    CastlingRights castling()   const { return castling_; }
    Square    ep_square()       const { return ep_square_; }
    int       halfmove_clock()  const { return halfmove_; }
    int       fullmove_number() const { return fullmove_; }
    uint64_t  hash()            const { return hash_; }

    Piece     piece_on(Square s)   const { return board_[s]; }
    Bitboard  pieces(Color c)      const { return occupied_[c]; }
    Bitboard  pieces()             const { return occupied_[2]; }
    Bitboard  pieces(Color c, PieceType pt) const { return pieces_bb_[c][pt]; }
    Bitboard  pieces(PieceType pt) const {
        return pieces_bb_[WHITE][pt] | pieces_bb_[BLACK][pt];
    }

    Square king_square(Color c) const {
        return BB::lsb(pieces_bb_[c][KING]);
    }

    // --- Move execution ---
    void make_move(Move m, UndoInfo& undo);
    void unmake_move(Move m, const UndoInfo& undo);

    // --- Display ---
    void print() const;

private:
    Bitboard pieces_bb_[COLOR_NB][PIECE_TYPE_NB]; // [color][piece_type]
    Bitboard occupied_[3];                          // [WHITE], [BLACK], [BOTH]
    Piece    board_[SQUARE_NB];                     // Mailbox

    Color          side_;
    CastlingRights castling_;
    Square         ep_square_;
    int            halfmove_;
    int            fullmove_;
    uint64_t       hash_;

    // Internal helpers
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    void recompute_hash();
};
