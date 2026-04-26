#include "engine_api.h"
#include "bitboard.h"
#include "position.h"
#include "nn_eval.h"
#include "search.h"

std::string get_best_move(const std::string& fen, int depth, const std::string& model_path) {
    static bool bb_initialized = false;
    if (!bb_initialized) {
        BB::init();
        bb_initialized = true;
    }

    Position pos;
    pos.set_fen(fen);

    NNEvaluator evaluator;
    evaluator.load_model(model_path);

    AlphaBetaSearch search(evaluator);
    Move best = search.get_best_move(pos, depth);

    return (best != MOVE_NONE) ? best.to_string() : "0000";
}
