#pragma once

#include "types.h"
#include "position.h"

#include <string>
#include <vector>
#include <memory>

// ============================================================
// Neural Network Evaluator using ONNX Runtime
// ============================================================

// Forward declare opaque ONNX Runtime types
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
}

class NNEvaluator {
public:
    NNEvaluator();
    ~NNEvaluator();

    // Load an ONNX model from disk
    bool load_model(const std::string& model_path);

    // Evaluate a position, returns score in [-1, 1]
    // +1 = white winning, -1 = black winning
    float evaluate(const Position& pos);

    // Check if a model is loaded
    bool is_loaded() const { return loaded_; }

    // Simple material-based evaluation fallback
    static float material_eval(const Position& pos);

private:
    bool loaded_ = false;

    // ONNX Runtime handles (opaque pointers to avoid header dependency)
    struct OrtState;
    std::unique_ptr<OrtState> ort_state_;

    // Encode board to the flattened (8, 8, 12) float array expected by Keras/ONNX.
    // If black is to move, board is rotated and colors are swapped so the network
    // always sees "side to move" as white perspective.
    void encode_board(const Position& pos, float* output) const;

    // Encode 8 metadata features:
    // [turn, wK, wQ, bK, bQ, ep_present, ep_file_norm, in_check]
    void encode_metadata(const Position& pos, float* output) const;
};
