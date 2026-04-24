#include "movegen.h"
#include "bitboard.h"

// ============================================================
// Attack detection
// ============================================================

bool is_square_attacked(const Position& pos, Square sq, Color by) {
    // Knight attacks
    if (BB::KnightAttacks[sq] & pos.pieces(by, KNIGHT))
        return true;

    // Pawn attacks (from sq's perspective, can 'by' color's pawns attack sq?)
    if (BB::PawnAttacks[~by][sq] & pos.pieces(by, PAWN))
        return true;

    // King attacks
    if (BB::KingAttacks[sq] & pos.pieces(by, KING))
        return true;

    Bitboard occ = pos.pieces();

    // Bishop/Queen (diagonal)
    if (BB::bishop_attacks(sq, occ) & (pos.pieces(by, BISHOP) | pos.pieces(by, QUEEN)))
        return true;

    // Rook/Queen (orthogonal)
    if (BB::rook_attacks(sq, occ) & (pos.pieces(by, ROOK) | pos.pieces(by, QUEEN)))
        return true;

    return false;
}

// ============================================================
// Pseudo-legal move generation helpers
// ============================================================

namespace {

// Add pawn moves, handling promotions automatically
void add_pawn_moves(MoveList& list, Square from, Bitboard targets, bool promotion) {
    while (targets) {
        Square to = BB::pop_lsb(targets);
        if (promotion) {
            list.add(Move::make_promotion(from, to, QUEEN));
            list.add(Move::make_promotion(from, to, ROOK));
            list.add(Move::make_promotion(from, to, BISHOP));
            list.add(Move::make_promotion(from, to, KNIGHT));
        } else {
            list.add(Move::make(from, to));
        }
    }
}

// Generate all pseudo-legal pawn moves
void generate_pawn_moves(const Position& pos, MoveList& list) {
    Color us   = pos.side_to_move();
    Color them = ~us;
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard occ       = pos.pieces();
    Bitboard enemies   = pos.pieces(them);

    Bitboard promo_rank = (us == WHITE) ? BB::RANK_8_BB : BB::RANK_1_BB;
    Bitboard third_rank = (us == WHITE) ? BB::RANK_3_BB : BB::RANK_6_BB;
    int push_dir = (us == WHITE) ? 8 : -8;

    while (our_pawns) {
        Square from = BB::pop_lsb(our_pawns);
        Bitboard from_bb = BB::square_bb(from);
        bool on_promo = false;

        // Single push
        Square push_sq = from + push_dir;
        if (push_sq >= A1 && push_sq <= H8 && !(occ & BB::square_bb(push_sq))) {
            on_promo = BB::square_bb(push_sq) & promo_rank;
            add_pawn_moves(list, from, BB::square_bb(push_sq), on_promo);

            // Double push (only if single push was possible)
            if (!on_promo && (from_bb & ((us == WHITE) ? BB::RANK_2_BB : BB::RANK_7_BB))) {
                Square double_sq = push_sq + push_dir;
                if (!(occ & BB::square_bb(double_sq))) {
                    list.add(Move::make(from, double_sq));
                }
            }
        }

        // Captures
        Bitboard captures = BB::PawnAttacks[us][from] & enemies;
        Bitboard promo_captures = captures & promo_rank;
        Bitboard normal_captures = captures & ~promo_rank;

        while (promo_captures) {
            Square to = BB::pop_lsb(promo_captures);
            list.add(Move::make_promotion(from, to, QUEEN));
            list.add(Move::make_promotion(from, to, ROOK));
            list.add(Move::make_promotion(from, to, BISHOP));
            list.add(Move::make_promotion(from, to, KNIGHT));
        }

        while (normal_captures) {
            Square to = BB::pop_lsb(normal_captures);
            list.add(Move::make(from, to));
        }

        // En passant
        Square ep = pos.ep_square();
        if (ep != NO_SQUARE && (BB::PawnAttacks[us][from] & BB::square_bb(ep))) {
            list.add(Move::make_en_passant(from, ep));
        }
    }
}

// Generate all pseudo-legal knight moves
void generate_knight_moves(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    Bitboard knights = pos.pieces(us, KNIGHT);
    Bitboard friendly = pos.pieces(us);

    while (knights) {
        Square from = BB::pop_lsb(knights);
        Bitboard targets = BB::KnightAttacks[from] & ~friendly;
        while (targets) {
            Square to = BB::pop_lsb(targets);
            list.add(Move::make(from, to));
        }
    }
}

// Generate all pseudo-legal bishop moves
void generate_bishop_moves(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    Bitboard bishops = pos.pieces(us, BISHOP);
    Bitboard friendly = pos.pieces(us);
    Bitboard occ = pos.pieces();

    while (bishops) {
        Square from = BB::pop_lsb(bishops);
        Bitboard targets = BB::bishop_attacks(from, occ) & ~friendly;
        while (targets) {
            Square to = BB::pop_lsb(targets);
            list.add(Move::make(from, to));
        }
    }
}

// Generate all pseudo-legal rook moves
void generate_rook_moves(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    Bitboard rooks = pos.pieces(us, ROOK);
    Bitboard friendly = pos.pieces(us);
    Bitboard occ = pos.pieces();

    while (rooks) {
        Square from = BB::pop_lsb(rooks);
        Bitboard targets = BB::rook_attacks(from, occ) & ~friendly;
        while (targets) {
            Square to = BB::pop_lsb(targets);
            list.add(Move::make(from, to));
        }
    }
}

// Generate all pseudo-legal queen moves
void generate_queen_moves(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    Bitboard queens = pos.pieces(us, QUEEN);
    Bitboard friendly = pos.pieces(us);
    Bitboard occ = pos.pieces();

    while (queens) {
        Square from = BB::pop_lsb(queens);
        Bitboard targets = BB::queen_attacks(from, occ) & ~friendly;
        while (targets) {
            Square to = BB::pop_lsb(targets);
            list.add(Move::make(from, to));
        }
    }
}

// Generate king moves including castling
void generate_king_moves(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    Square king_sq = pos.king_square(us);
    Bitboard friendly = pos.pieces(us);

    // Normal king moves
    Bitboard targets = BB::KingAttacks[king_sq] & ~friendly;
    while (targets) {
        Square to = BB::pop_lsb(targets);
        list.add(Move::make(king_sq, to));
    }

    // Castling
    Bitboard occ = pos.pieces();
    Color them = ~us;

    if (us == WHITE) {
        // White kingside: E1-G1, need F1 & G1 clear, E1/F1/G1 not attacked
        if ((pos.castling() & WHITE_OO) &&
            !(occ & (BB::square_bb(F1) | BB::square_bb(G1))) &&
            !is_square_attacked(pos, E1, them) &&
            !is_square_attacked(pos, F1, them) &&
            !is_square_attacked(pos, G1, them))
        {
            list.add(Move::make_castling(E1, G1));
        }
        // White queenside: E1-C1, need B1/C1/D1 clear, E1/D1/C1 not attacked
        if ((pos.castling() & WHITE_OOO) &&
            !(occ & (BB::square_bb(D1) | BB::square_bb(C1) | BB::square_bb(B1))) &&
            !is_square_attacked(pos, E1, them) &&
            !is_square_attacked(pos, D1, them) &&
            !is_square_attacked(pos, C1, them))
        {
            list.add(Move::make_castling(E1, C1));
        }
    } else {
        // Black kingside
        if ((pos.castling() & BLACK_OO) &&
            !(occ & (BB::square_bb(F8) | BB::square_bb(G8))) &&
            !is_square_attacked(pos, E8, them) &&
            !is_square_attacked(pos, F8, them) &&
            !is_square_attacked(pos, G8, them))
        {
            list.add(Move::make_castling(E8, G8));
        }
        // Black queenside
        if ((pos.castling() & BLACK_OOO) &&
            !(occ & (BB::square_bb(D8) | BB::square_bb(C8) | BB::square_bb(B8))) &&
            !is_square_attacked(pos, E8, them) &&
            !is_square_attacked(pos, D8, them) &&
            !is_square_attacked(pos, C8, them))
        {
            list.add(Move::make_castling(E8, C8));
        }
    }
}

} // anonymous namespace

// ============================================================
// Legal move generation
// ============================================================

void generate_moves(const Position& pos, MoveList& list) {
    list.count = 0;

    // Generate all pseudo-legal moves
    MoveList pseudo;
    pseudo.count = 0;

    generate_pawn_moves(pos, pseudo);
    generate_knight_moves(pos, pseudo);
    generate_bishop_moves(pos, pseudo);
    generate_rook_moves(pos, pseudo);
    generate_queen_moves(pos, pseudo);
    generate_king_moves(pos, pseudo);

    // Filter: keep only moves that don't leave our king in check
    Color us = pos.side_to_move();

    for (int i = 0; i < pseudo.count; ++i) {
        Move m = pseudo[i];

        // We need a mutable copy to make/unmake
        Position pos_copy = pos;
        UndoInfo undo;
        pos_copy.make_move(m, undo);

        // After making the move, side_to_move is now the opponent.
        // Check if OUR king (the side that just moved) is in check.
        Square king_sq = pos_copy.king_square(us);
        if (!is_square_attacked(pos_copy, king_sq, ~us)) {
            list.add(m);
        }
    }
}
