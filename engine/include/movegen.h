#pragma once

#include "position.h"
#include "move.h"

// ============================================================
// Move generation interface
// ============================================================

// Generate all legal moves for the current position
void generate_moves(const Position& pos, MoveList& list);

// Check if a square is attacked by the given side
bool is_square_attacked(const Position& pos, Square sq, Color by);
