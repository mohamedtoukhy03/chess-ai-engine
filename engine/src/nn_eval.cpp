#include "nn_eval.h"
#include "bitboard.h"

#include <cmath>
#include <iostream>
#include <array>

// ============================================================
// Conditional ONNX Runtime support
// If ONNX Runtime is available, use it; otherwise fall back to material eval
// ============================================================

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

struct NNEvaluator::OrtState {
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;

    OrtState()
        : env(ORT_LOGGING_LEVEL_WARNING, "ChessEngine")
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {}
};

NNEvaluator::NNEvaluator() : ort_state_(std::make_unique<OrtState>()) {}
NNEvaluator::~NNEvaluator() = default;

bool NNEvaluator::load_model(const std::string& model_path) {
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        ort_state_->session = std::make_unique<Ort::Session>(
            ort_state_->env, model_path.c_str(), opts
        );

        loaded_ = true;
        std::cout << "info string NN model loaded: " << model_path << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "info string Failed to load ONNX model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

float NNEvaluator::evaluate(const Position& pos) {
    if (!loaded_) {
        return material_eval(pos);
    }

    // Encode board
    std::array<float, 12 * 8 * 8> input_data{};
    encode_board(pos, input_data.data());

    // Create input tensor
    std::array<int64_t, 4> input_shape = {1, 12, 8, 8};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_state_->memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );

    // Run inference
    const char* input_names[] = {"board"};
    const char* output_names[] = {"eval"};

    auto output_tensors = ort_state_->session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float eval = output_tensors[0].GetTensorData<float>()[0];

    // Flip evaluation if black to move (NN always evaluates from white's perspective)
    if (pos.side_to_move() == BLACK) {
        eval = -eval;
    }

    return eval;
}

#else
// No ONNX Runtime — pure material evaluation fallback

struct NNEvaluator::OrtState {};

NNEvaluator::NNEvaluator() : ort_state_(std::make_unique<OrtState>()) {}
NNEvaluator::~NNEvaluator() = default;

bool NNEvaluator::load_model(const std::string& model_path) {
    (void)model_path;
    std::cout << "info string ONNX Runtime not available, using material evaluation" << std::endl;
    loaded_ = false;
    return false;
}

float NNEvaluator::evaluate(const Position& pos) {
    return material_eval(pos);
}

#endif // USE_ONNXRUNTIME

// ============================================================
// Board encoding (shared between ONNX and fallback paths)
// ============================================================

void NNEvaluator::encode_board(const Position& pos, float* output) const {
    // Output layout: 12 planes of 8×8
    // Planes 0-5:  White P, N, B, R, Q, K
    // Planes 6-11: Black P, N, B, R, Q, K
    std::fill(output, output + 12 * 64, 0.0f);

    for (int c = 0; c < COLOR_NB; ++c) {
        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
            int plane = c * 6 + pt;
            Bitboard bb = pos.pieces(Color(c), PieceType(pt));
            while (bb) {
                Square sq = BB::pop_lsb(bb);
                int rank = square_rank(sq);
                int file = square_file(sq);
                output[plane * 64 + rank * 8 + file] = 1.0f;
            }
        }
    }
}

// ============================================================
// Material evaluation fallback
// ============================================================

float NNEvaluator::material_eval(const Position& pos) {
    constexpr int PIECE_VALUES[PIECE_TYPE_NB] = {
        100,  // Pawn
        320,  // Knight
        330,  // Bishop
        500,  // Rook
        900,  // Queen
        0     // King (not counted)
    };

    int score = 0;

    for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
        score += BB::popcount(pos.pieces(WHITE, PieceType(pt))) * PIECE_VALUES[pt];
        score -= BB::popcount(pos.pieces(BLACK, PieceType(pt))) * PIECE_VALUES[pt];
    }

    // Normalize to [-1, 1] range using tanh
    float normalized = std::tanh(score / 400.0f);

    // Flip if black to move (return from current player's perspective)
    if (pos.side_to_move() == BLACK)
        normalized = -normalized;

    return normalized;
}
