#pragma once

#include "position.h"
#include "move.h"
#include "nn_eval.h"

#include <chrono>

class AlphaBetaSearch {
public:
    AlphaBetaSearch(NNEvaluator& evaluator);

    // Search for the best move given a time limit in milliseconds
    Move search(Position& pos, int time_ms);

private:
    NNEvaluator& evaluator_;
    
    int nodes_ = 0;
    int seldepth_ = 0;
    std::chrono::time_point<std::chrono::steady_clock> deadline_;
    bool time_over_ = false;

    // Checks if the search should abort due to time limits
    void check_time();

    // Recursive Alpha-Beta function
    float alpha_beta(Position& pos, int depth, float alpha, float beta, int ply);
    
    // Sort moves to dramatically speed up pruning
    void sort_moves(Position& pos, MoveList& moves);
};
