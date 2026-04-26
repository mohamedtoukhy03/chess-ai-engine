#include "search.h"
#include "movegen.h"
#include <iostream>
#include <algorithm>

constexpr float MATE_SCORE = 1000.0f;
constexpr float DRAW_SCORE = 0.0f;

AlphaBetaSearch::AlphaBetaSearch(NNEvaluator& evaluator)
    : evaluator_(evaluator) {}

void AlphaBetaSearch::check_time() {
    if (nodes_ % 1024 == 0) {
        if (std::chrono::steady_clock::now() > deadline_) {
            time_over_ = true;
        }
    }
}

void AlphaBetaSearch::sort_moves(Position& pos, MoveList& moves) {
    // Simple move ordering: captures and promotions first
    int scores[256] = {0};
    
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves[i];
        if (pos.piece_on(m.to()) != NO_PIECE) {
            scores[i] += 100; // Capture bonus
        }
        if (m.is_promotion()) {
            scores[i] += 50;  // Promo bonus
        }
    }
    
    // Insertion sort based on score
    for (int i = 1; i < moves.count; ++i) {
        Move curr_m = moves[i];
        int curr_s = scores[i];
        int j = i - 1;
        
        while (j >= 0 && scores[j] < curr_s) {
            moves[j + 1] = moves[j];
            scores[j + 1] = scores[j];
            j--;
        }
        moves[j + 1] = curr_m;
        scores[j + 1] = curr_s;
    }
}

float AlphaBetaSearch::alpha_beta(Position& pos, int depth, float alpha, float beta, int ply) {
    check_time();
    if (time_over_) return 0.0f;
    
    if (ply > seldepth_) seldepth_ = ply;
    
    // Draw detection (50-move rule)
    if (ply > 0 && pos.halfmove_clock() >= 100) {
        return DRAW_SCORE;
    }
    
    MoveList moves;
    generate_moves(pos, moves);
    
    // Terminal states
    if (moves.count == 0) {
        Square king_sq = pos.king_square(pos.side_to_move());
        if (is_square_attacked(pos, king_sq, ~pos.side_to_move())) {
            return -(MATE_SCORE - ply * 0.01f); // Checkmate
        }
        return DRAW_SCORE; // Stalemate
    }
    
    // Leaf node: evaluate state
    if (depth <= 0) {
        nodes_++;
        return evaluator_.evaluate(pos);
    }
    
    sort_moves(pos, moves);
    
    float best_score = -MATE_SCORE * 10;
    
    for (int i = 0; i < moves.count; ++i) {
        UndoInfo undo;
        pos.make_move(moves[i], undo);
        
        float score = -alpha_beta(pos, depth - 1, -beta, -alpha, ply + 1);
        
        pos.unmake_move(moves[i], undo);
        
        if (time_over_) return 0.0f;
        
        if (score > best_score) {
            best_score = score;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            break; // Beta Cutoff (pruning)
        }
    }
    
    return best_score;
}

Move AlphaBetaSearch::search(Position& pos, int time_ms) {
    auto start = std::chrono::steady_clock::now();
    deadline_ = start + std::chrono::milliseconds(time_ms);
    time_over_ = false;
    nodes_ = 0;
    
    Move best_move = MOVE_NONE;
    int max_depth = 100;
    
    for (int depth = 1; depth <= max_depth; ++depth) {
        seldepth_ = 0;
        float alpha = -MATE_SCORE * 10;
        float beta = MATE_SCORE * 10;
        
        Move iter_best = MOVE_NONE;
        float best_score = -MATE_SCORE * 10;
        
        MoveList moves;
        generate_moves(pos, moves);
        sort_moves(pos, moves);
        
        // Principal Variation (PV) ordering: push previous best move to front
        if (best_move != MOVE_NONE) {
            for (int i = 0; i < moves.count; ++i) {
                if (moves[i] == best_move) {
                    // Swap to front
                    Move temp = moves[0];
                    moves[0] = moves[i];
                    moves[i] = temp;
                    break;
                }
            }
        }
        
        for (int i = 0; i < moves.count; ++i) {
            UndoInfo undo;
            pos.make_move(moves[i], undo);
            
            float score = -alpha_beta(pos, depth - 1, -beta, -alpha, 1);
            
            pos.unmake_move(moves[i], undo);
            
            if (time_over_) break;
            
            if (score > best_score) {
                best_score = score;
                iter_best = moves[i];
            }
            if (score > alpha) {
                alpha = score;
            }
        }
        
        if (time_over_) {
            break; // Throw away partial depth search metrics
        }
        
        best_move = iter_best;
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
        ).count();
        
        std::cout << "info depth " << depth 
                  << " seldepth " << seldepth_
                  << " score cp " << static_cast<int>(best_score * 100) 
                  << " nodes " << nodes_ 
                  << " time " << elapsed 
                  << (elapsed > 0 ? " nps " + std::to_string(nodes_ * 1000 / elapsed) : "")
                  << " pv " << best_move.to_string() 
                  << std::endl;
                  
        // If mate is forced, break
        if (best_score > MATE_SCORE - 100.0f) {
            break;
        }
    }
    
    return best_move;
}

Move AlphaBetaSearch::get_best_move(Position& pos, int depth) {
    nodes_ = 0;
    seldepth_ = 0;
    time_over_ = false;
    deadline_ = std::chrono::steady_clock::time_point::max();

    Move best_move = MOVE_NONE;
    float alpha = -MATE_SCORE * 10;
    const float beta = MATE_SCORE * 10;
    float best_score = -MATE_SCORE * 10;

    MoveList moves;
    generate_moves(pos, moves);
    sort_moves(pos, moves);

    for (int i = 0; i < moves.count; ++i) {
        UndoInfo undo;
        pos.make_move(moves[i], undo);
        float score = -alpha_beta(pos, depth - 1, -beta, -alpha, 1);
        pos.unmake_move(moves[i], undo);

        if (score > best_score) {
            best_score = score;
            best_move = moves[i];
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    std::cout << "info depth " << depth
              << " score cp " << static_cast<int>(best_score * 100)
              << " nodes " << nodes_
              << " pv " << (best_move != MOVE_NONE ? best_move.to_string() : "0000")
              << std::endl;

    return best_move;
}
