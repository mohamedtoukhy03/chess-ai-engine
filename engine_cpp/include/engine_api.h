#pragma once

#include <string>

// High-level deployment API:
// Returns best move in UCI notation (e.g., "e2e4"), or "0000" if none.
std::string get_best_move(const std::string& fen, int depth, const std::string& model_path);
