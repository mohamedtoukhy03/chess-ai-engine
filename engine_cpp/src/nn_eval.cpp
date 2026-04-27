#include "nn_eval.h"
#include "bitboard.h"
#include "movegen.h"

#include <cmath>
#include <iostream>
#include <array>
#include <vector>

using namespace std;

// ============================================================
// ONNX Runtime Integration
// ============================================================

#include <onnxruntime_cxx_api.h>

struct NNEvaluator::OrtState {
    Ort::Env env;
    unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    vector<string> input_names_str;
    vector<const char*> input_names;
    vector<string> output_names_str;
    vector<const char*> output_names;

    OrtState()
        : env(ORT_LOGGING_LEVEL_WARNING, "ChessEngine")
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {}
};

NNEvaluator::NNEvaluator() : ort_state_(make_unique<OrtState>()) {}
NNEvaluator::~NNEvaluator() = default;

bool NNEvaluator::load_model(const string& model_path) {
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        ort_state_->session = make_unique<Ort::Session>(
            ort_state_->env, model_path.c_str(), opts
        );

        // Cache model IO names (handles both 1-input and 2-input exports).
        ort_state_->input_names_str.clear();
        ort_state_->input_names.clear();
        ort_state_->output_names_str.clear();
        ort_state_->output_names.clear();

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = ort_state_->session->GetInputCount();
        const size_t output_count = ort_state_->session->GetOutputCount();

        ort_state_->input_names_str.reserve(input_count);
        ort_state_->input_names.reserve(input_count);
        ort_state_->output_names_str.reserve(output_count);
        ort_state_->output_names.reserve(output_count);

        for (size_t i = 0; i < input_count; ++i) {
            auto name = ort_state_->session->GetInputNameAllocated(i, allocator);
            ort_state_->input_names_str.emplace_back(name.get());
            ort_state_->input_names.push_back(ort_state_->input_names_str.back().c_str());
        }
        for (size_t i = 0; i < output_count; ++i) {
            auto name = ort_state_->session->GetOutputNameAllocated(i, allocator);
            ort_state_->output_names_str.emplace_back(name.get());
            ort_state_->output_names.push_back(ort_state_->output_names_str.back().c_str());
        }

        loaded_ = true;
        cout << "info string NN model loaded: " << model_path << endl;
        return true;
    } catch (const Ort::Exception& e) {
        cerr << "info string Failed to load ONNX model: " << e.what() << endl;
        loaded_ = false;
        return false;
    }
}

float NNEvaluator::evaluate(const Position& pos) {
    if (!loaded_) {
        cerr << "Engine attempted to evaluate without a loaded AI model" << endl;
        return 0.0f; // Safety fallback since search should abort first
    }

    // Encode inputs
    array<float, 8 * 8 * 12> board_data{};
    array<float, 8> meta_data{};
    encode_board(pos, board_data.data());
    encode_metadata(pos, meta_data.data());

    array<int64_t, 4> board_shape = {1, 8, 8, 12}; // NHWC
    array<int64_t, 2> meta_shape = {1, 8};
    Ort::Value board_tensor = Ort::Value::CreateTensor<float>(
        ort_state_->memory_info, board_data.data(), board_data.size(),
        board_shape.data(), board_shape.size()
    );
    Ort::Value meta_tensor = Ort::Value::CreateTensor<float>(
        ort_state_->memory_info, meta_data.data(), meta_data.size(),
        meta_shape.data(), meta_shape.size()
    );

    vector<Ort::Value> inputs;
    inputs.emplace_back(move(board_tensor));

    // If the exported model has a metadata input, pass it too.
    if (ort_state_->input_names.size() >= 2) {
        inputs.emplace_back(move(meta_tensor));
    }

    auto output_tensors = ort_state_->session->Run(
        Ort::RunOptions{nullptr},
        ort_state_->input_names.data(), inputs.data(), inputs.size(),
        ort_state_->output_names.data(), ort_state_->output_names.size()
    );

    float nn_score = output_tensors[0].GetTensorData<float>()[0];
    float mat_score = material_eval(pos);
    
    // Add raw material evaluation as a bias to the Neural Network
    return nn_score + (mat_score * 0.5f); // Combining AI and piece importance
}

// ============================================================
// Board encoding (shared between ONNX and fallback paths)
// ============================================================

void NNEvaluator::encode_board(const Position& pos, float* output) const {
    // Output layout: flattened (rank, file, channel) for shape (8, 8, 12), NHWC.
    // Rank 0 corresponds to board rank 8 to match the Python FEN parser.
    fill(output, output + (8 * 8 * 12), 0.0f);

    const bool flip_perspective = (pos.side_to_move() == BLACK);

    for (int sq_i = 0; sq_i < SQUARE_NB; ++sq_i) {
        Piece p = pos.piece_on(Square(sq_i));
        if (p == NO_PIECE) {
            continue;
        }

        int file = square_file(Square(sq_i));
        int rank = square_rank(Square(sq_i)); // 0 = rank1 ... 7 = rank8
        int channel = static_cast<int>(p);

        if (flip_perspective) {
            // Rotate board 180 and swap colors to canonical side-to-move perspective.
            file = 7 - file;
            rank = 7 - rank;
            channel = (channel < 6) ? (channel + 6) : (channel - 6);
        }

        // Tensor rank index follows FEN order (top rank first).
        const int tensor_rank = 7 - rank;
        const int idx = ((tensor_rank * 8 + file) * 12) + channel;
        output[idx] = 1.0f;
    }
}

void NNEvaluator::encode_metadata(const Position& pos, float* output) const {
    const bool flip_perspective = (pos.side_to_move() == BLACK);
    const CastlingRights c = pos.castling();

    // Turn after canonicalization is always "white to move".
    output[0] = 1.0f;

    // Castling flags in canonical perspective (side-to-move first).
    if (!flip_perspective) {
        output[1] = (c & WHITE_OO)  ? 1.0f : 0.0f;
        output[2] = (c & WHITE_OOO) ? 1.0f : 0.0f;
        output[3] = (c & BLACK_OO)  ? 1.0f : 0.0f;
        output[4] = (c & BLACK_OOO) ? 1.0f : 0.0f;
    } else {
        output[1] = (c & BLACK_OO)  ? 1.0f : 0.0f;
        output[2] = (c & BLACK_OOO) ? 1.0f : 0.0f;
        output[3] = (c & WHITE_OO)  ? 1.0f : 0.0f;
        output[4] = (c & WHITE_OOO) ? 1.0f : 0.0f;
    }

    // En passant: presence + normalized file.
    const Square ep = pos.ep_square();
    if (ep != NO_SQUARE) {
        int ep_file = square_file(ep);
        if (flip_perspective) {
            ep_file = 7 - ep_file;
        }
        output[5] = 1.0f;
        output[6] = static_cast<float>(ep_file) / 7.0f;
    } else {
        output[5] = 0.0f;
        output[6] = 0.0f;
    }

    // In-check flag for side to move.
    const Square ksq = pos.king_square(pos.side_to_move());
    output[7] = is_square_attacked(pos, ksq, ~pos.side_to_move()) ? 1.0f : 0.0f;
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
    float normalized = tanh(score / 400.0f);

    // Flip if black to move (return from current player's perspective)
    if (pos.side_to_move() == BLACK)
        normalized = -normalized;

    return normalized;
}
