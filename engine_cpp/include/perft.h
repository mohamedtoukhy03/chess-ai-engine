#pragma once

#include "position.h"
#include <cstdint>

// Count leaf nodes at the given depth (for validation)
uint64_t perft(Position& pos, int depth);

// Divide: perft per root move (for debugging mismatches)
void divide(Position& pos, int depth);
