#include "perft.h"
#include "movegen.h"

#include <iostream>
#include <iomanip>

// ============================================================
// Perft: recursive leaf node count
// ============================================================

uint64_t perft(Position& pos, int depth) {
    if (depth == 0)
        return 1;

    MoveList moves;
    generate_moves(pos, moves);

    if (depth == 1)
        return moves.count; // Bulk counting optimization

    uint64_t nodes = 0;
    for (int i = 0; i < moves.count; ++i) {
        UndoInfo undo;
        pos.make_move(moves[i], undo);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
    }

    return nodes;
}

// ============================================================
// Divide: per-root-move node count (for isolating bugs)
// ============================================================

void divide(Position& pos, int depth) {
    MoveList moves;
    generate_moves(pos, moves);

    uint64_t total = 0;

    for (int i = 0; i < moves.count; ++i) {
        UndoInfo undo;
        pos.make_move(moves[i], undo);
        uint64_t nodes = (depth <= 1) ? 1 : perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);

        std::cout << moves[i].to_string() << ": " << nodes << "\n";
        total += nodes;
    }

    std::cout << "\nTotal: " << total << "\n";
    std::cout << "Moves: " << moves.count << "\n";
}
