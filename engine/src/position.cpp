#include "position.h"

#include <iostream>
#include <sstream>
#include <cstring>

// ============================================================
// Internal piece manipulation
// ============================================================

void Position::put_piece(Piece p, Square s) {
    Color c = piece_color(p);
    PieceType pt = piece_type(p);

    board_[s] = p;
    pieces_bb_[c][pt] |= BB::square_bb(s);
    occupied_[c]      |= BB::square_bb(s);
    occupied_[2]      |= BB::square_bb(s);

    hash_ ^= BB::Zobrist::PieceSquare[p][s];
}

void Position::remove_piece(Square s) {
    Piece p = board_[s];
    assert(p != NO_PIECE);
    Color c = piece_color(p);
    PieceType pt = piece_type(p);

    board_[s] = NO_PIECE;
    pieces_bb_[c][pt] &= ~BB::square_bb(s);
    occupied_[c]      &= ~BB::square_bb(s);
    occupied_[2]      &= ~BB::square_bb(s);

    hash_ ^= BB::Zobrist::PieceSquare[p][s];
}

void Position::move_piece(Square from, Square to) {
    Piece p = board_[from];
    assert(p != NO_PIECE);
    assert(board_[to] == NO_PIECE);

    Color c = piece_color(p);
    PieceType pt = piece_type(p);
    Bitboard fromto = BB::square_bb(from) | BB::square_bb(to);

    board_[from] = NO_PIECE;
    board_[to]   = p;
    pieces_bb_[c][pt] ^= fromto;
    occupied_[c]      ^= fromto;
    occupied_[2]      ^= fromto;

    hash_ ^= BB::Zobrist::PieceSquare[p][from];
    hash_ ^= BB::Zobrist::PieceSquare[p][to];
}

// ============================================================
// Constructor & start position
// ============================================================

Position::Position() {
    std::memset(pieces_bb_, 0, sizeof(pieces_bb_));
    std::memset(occupied_, 0, sizeof(occupied_));
    std::fill(std::begin(board_), std::end(board_), NO_PIECE);

    side_      = WHITE;
    castling_  = NO_CASTLING;
    ep_square_ = NO_SQUARE;
    halfmove_  = 0;
    fullmove_  = 1;
    hash_      = 0;
}

void Position::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// ============================================================
// FEN parsing and generation
// ============================================================

void Position::set_fen(const std::string& fen) {
    // Reset
    std::memset(pieces_bb_, 0, sizeof(pieces_bb_));
    std::memset(occupied_, 0, sizeof(occupied_));
    std::fill(std::begin(board_), std::end(board_), NO_PIECE);
    hash_ = 0;

    std::istringstream ss(fen);
    std::string token;

    // 1. Piece placement
    ss >> token;
    int rank = 7, file = 0;
    for (char c : token) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            Piece p = char_to_piece(c);
            if (p != NO_PIECE) {
                put_piece(p, make_square(file, rank));
                file++;
            }
        }
    }

    // 2. Side to move
    ss >> token;
    side_ = (token == "w") ? WHITE : BLACK;
    if (side_ == BLACK) hash_ ^= BB::Zobrist::SideToMove;

    // 3. Castling rights
    ss >> token;
    castling_ = NO_CASTLING;
    for (char c : token) {
        switch (c) {
            case 'K': castling_ |= WHITE_OO;  break;
            case 'Q': castling_ |= WHITE_OOO; break;
            case 'k': castling_ |= BLACK_OO;  break;
            case 'q': castling_ |= BLACK_OOO; break;
            default: break;
        }
    }
    hash_ ^= BB::Zobrist::Castling[castling_];

    // 4. En passant
    ss >> token;
    ep_square_ = string_to_square(token);
    if (ep_square_ != NO_SQUARE)
        hash_ ^= BB::Zobrist::EnPassant[square_file(ep_square_)];

    // 5. Halfmove clock
    if (ss >> token)
        halfmove_ = std::stoi(token);
    else
        halfmove_ = 0;

    // 6. Fullmove number
    if (ss >> token)
        fullmove_ = std::stoi(token);
    else
        fullmove_ = 1;
}

std::string Position::to_fen() const {
    std::string fen;

    // 1. Piece placement
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board_[make_square(f, r)];
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) {
                    fen += std::to_string(empty);
                    empty = 0;
                }
                fen += piece_to_char(p);
            }
        }
        if (empty > 0)
            fen += std::to_string(empty);
        if (r > 0) fen += '/';
    }

    // 2. Side to move
    fen += (side_ == WHITE) ? " w " : " b ";

    // 3. Castling
    if (castling_ == NO_CASTLING) {
        fen += '-';
    } else {
        if (castling_ & WHITE_OO)  fen += 'K';
        if (castling_ & WHITE_OOO) fen += 'Q';
        if (castling_ & BLACK_OO)  fen += 'k';
        if (castling_ & BLACK_OOO) fen += 'q';
    }

    // 4. En passant
    fen += ' ';
    fen += square_to_string(ep_square_);

    // 5-6. Halfmove and fullmove
    fen += ' ' + std::to_string(halfmove_);
    fen += ' ' + std::to_string(fullmove_);

    return fen;
}

// ============================================================
// Make / Unmake move
// ============================================================

// Castling rights mask: when a piece moves from/to these squares,
// the corresponding castling rights are lost.
static CastlingRights castling_rights_mask(Square s) {
    // Revoke rights when rook or king moves from its original square,
    // or when a piece captures on a rook's original square
    switch (s) {
        case E1: return CastlingRights(~WHITE_CASTLING & 0xF);
        case A1: return CastlingRights(~WHITE_OOO & 0xF);
        case H1: return CastlingRights(~WHITE_OO  & 0xF);
        case E8: return CastlingRights(~BLACK_CASTLING & 0xF);
        case A8: return CastlingRights(~BLACK_OOO & 0xF);
        case H8: return CastlingRights(~BLACK_OO  & 0xF);
        default: return ALL_CASTLING;
    }
}

void Position::make_move(Move m, UndoInfo& undo) {
    // Save undo state
    undo.castling  = castling_;
    undo.ep_square = ep_square_;
    undo.halfmove  = halfmove_;
    undo.hash      = hash_;

    Square from = m.from();
    Square to   = m.to();
    Piece moving = board_[from];
    PieceType pt = piece_type(moving);

    assert(moving != NO_PIECE);
    assert(piece_color(moving) == side_);

    // Remove old en passant from hash
    if (ep_square_ != NO_SQUARE)
        hash_ ^= BB::Zobrist::EnPassant[square_file(ep_square_)];
    ep_square_ = NO_SQUARE;

    // Remove old castling from hash
    hash_ ^= BB::Zobrist::Castling[castling_];

    if (m.is_castling()) {
        // King move is encoded; we also need to move the rook
        undo.captured = NO_PIECE;

        // Determine rook squares based on king destination
        Square rook_from, rook_to;
        if (to > from) {
            // Kingside
            rook_from = (side_ == WHITE) ? H1 : H8;
            rook_to   = (side_ == WHITE) ? F1 : F8;
        } else {
            // Queenside
            rook_from = (side_ == WHITE) ? A1 : A8;
            rook_to   = (side_ == WHITE) ? D1 : D8;
        }

        move_piece(from, to);        // King
        move_piece(rook_from, rook_to); // Rook

        castling_ &= castling_rights_mask(from);
        halfmove_++;
    }
    else if (m.is_en_passant()) {
        // Captured pawn is on a different square than 'to'
        Square cap_sq = (side_ == WHITE) ? to - 8 : to + 8;
        undo.captured = board_[cap_sq];
        assert(piece_type(undo.captured) == PAWN);

        remove_piece(cap_sq);
        move_piece(from, to);

        halfmove_ = 0;
    }
    else if (m.is_promotion()) {
        undo.captured = board_[to];

        // Remove captured piece if any
        if (undo.captured != NO_PIECE)
            remove_piece(to);

        // Remove the pawn, place the promoted piece
        remove_piece(from);
        Piece promo_piece = make_piece(side_, m.promotion_type());
        put_piece(promo_piece, to);

        // Update castling rights (capture on rook square)
        castling_ &= castling_rights_mask(from);
        castling_ &= castling_rights_mask(to);
        halfmove_ = 0;
    }
    else {
        // Normal move
        undo.captured = board_[to];

        if (undo.captured != NO_PIECE)
            remove_piece(to);

        move_piece(from, to);

        // Update castling rights
        castling_ &= castling_rights_mask(from);
        castling_ &= castling_rights_mask(to);

        // Halfmove clock: reset on pawn move or capture
        if (pt == PAWN || undo.captured != NO_PIECE)
            halfmove_ = 0;
        else
            halfmove_++;

        // Double pawn push: set en passant square
        if (pt == PAWN) {
            int diff = int(to) - int(from);
            if (diff == 16 || diff == -16) {
                ep_square_ = Square((int(from) + int(to)) / 2);
                hash_ ^= BB::Zobrist::EnPassant[square_file(ep_square_)];
            }
        }
    }

    // Update castling hash
    hash_ ^= BB::Zobrist::Castling[castling_];

    // Flip side to move
    side_ = ~side_;
    hash_ ^= BB::Zobrist::SideToMove;

    // Increment fullmove after black moves
    if (side_ == WHITE)
        fullmove_++;
}

void Position::unmake_move(Move m, const UndoInfo& undo) {
    // Flip side back
    side_ = ~side_;

    Square from = m.from();
    Square to   = m.to();

    if (m.is_castling()) {
        // Undo king and rook moves
        Square rook_from, rook_to;
        if (to > from) {
            rook_from = (side_ == WHITE) ? H1 : H8;
            rook_to   = (side_ == WHITE) ? F1 : F8;
        } else {
            rook_from = (side_ == WHITE) ? A1 : A8;
            rook_to   = (side_ == WHITE) ? D1 : D8;
        }
        move_piece(rook_to, rook_from);
        move_piece(to, from);
    }
    else if (m.is_en_passant()) {
        // Move pawn back and restore captured pawn
        move_piece(to, from);
        Square cap_sq = (side_ == WHITE) ? to - 8 : to + 8;
        put_piece(undo.captured, cap_sq);
    }
    else if (m.is_promotion()) {
        // Remove promoted piece, restore pawn
        remove_piece(to);
        put_piece(make_piece(side_, PAWN), from);

        // Restore captured piece if any
        if (undo.captured != NO_PIECE)
            put_piece(undo.captured, to);
    }
    else {
        // Normal move: move piece back, restore capture
        move_piece(to, from);
        if (undo.captured != NO_PIECE)
            put_piece(undo.captured, to);
    }

    // Restore saved state
    castling_  = undo.castling;
    ep_square_ = undo.ep_square;
    halfmove_  = undo.halfmove;
    hash_      = undo.hash;

    if (side_ == BLACK)
        fullmove_--;
}

// ============================================================
// Recompute hash from scratch (for validation)
// ============================================================

void Position::recompute_hash() {
    hash_ = 0;
    for (int sq = 0; sq < SQUARE_NB; ++sq) {
        if (board_[sq] != NO_PIECE)
            hash_ ^= BB::Zobrist::PieceSquare[board_[sq]][sq];
    }
    if (side_ == BLACK)
        hash_ ^= BB::Zobrist::SideToMove;
    hash_ ^= BB::Zobrist::Castling[castling_];
    if (ep_square_ != NO_SQUARE)
        hash_ ^= BB::Zobrist::EnPassant[square_file(ep_square_)];
}

// ============================================================
// Print: ASCII board for debugging
// ============================================================

void Position::print() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r + 1) << " |";
        for (int f = 0; f < 8; ++f) {
            Piece p = board_[make_square(f, r)];
            char c = (p == NO_PIECE) ? ' ' : piece_to_char(p);
            std::cout << ' ' << c << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
    std::cout << "FEN: " << to_fen() << "\n";
    std::cout << "Hash: 0x" << std::hex << hash_ << std::dec << "\n\n";
}
