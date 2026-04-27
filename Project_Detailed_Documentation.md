# Comprehensive Guide: Neural Network Chess AI Engine

This document provides a deep, meticulous explanation of the University Neural Network Chess AI project. It breaks down the core concepts, the Deep Learning architecture, the training pipeline, and the high-performance C++ chess engine, including an in-depth code explanation of the critical engine components.

---

## 1. The Core Idea

At its heart, your project is a highly optimized "Engine" that actively plays chess. Traditional engines (like older versions of Stockfish) relied on hardcoded, "hand-crafted" rules to evaluate whether a position was good (e.g., "A Knight is worth 3 pawns," "castling is good," "doubled pawns are bad"). 

Your engine modernizes this approach by replacing hardcoded rules with a **Neural Network**. 

### How the Engine Works (Step-by-Step):
1. **Move Generation**: The engine generates all possible legal moves up to a certain number of moves ahead (depth) using **Bitboards** (super-fast 64-bit integer math).
2. **AI Evaluation**: Instead of counting piece values, it uses an ONNX-exported neural network to look at the resulting chess boards and predict a "score" (eval) representing the balance of the game and who is winning.
3. **Communication**: It communicates with standard chess GUIs (like Arena, Cute Chess, or Lichess) via the **UCI** (Universal Chess Interface) protocol.

---

## 2. The Neural Network Model

The model implemented in your Python code is an **SE-ResNet (Squeeze-and-Excitation Residual Network)**. It's built in TensorFlow/Keras and designed to understand the spatial complexities of a chess board.

### Inputs
It takes two distinct inputs simultaneously:
1. `board_in`: The actual chess board encoded as an $8 \times 8 \times 12$ matrix (8 rows, 8 columns, 12 channels for 6 white piece types + 6 black piece types). A `1` in a specific channel means a piece exists there.
2. `meta_in`: Extra contextual metadata (whose turn it is, castling rights, en passant availability, and whether the king is currently in check).

### Architecture
* **Stem**: Starts with a Convolutional 2D (Conv2D) stem to extract low-level features of the board.
* **Blocks**: Passes through numerous `res_se_block` structures. These are residual blocks that include a Squeeze-and-Excitation layer. 
  * *Why SE?* Squeeze-and-Excitation layers allow the network to "focus" on important piece channels automatically by weighting them, heavily boosting performance on spatial tasks like Chess.
* **Head**: The output is flattened, concatenated with the `meta_in` metadata, and passed through dense feed-forward layers to predict a single scalar floating-point value (`eval`), representing how good the board is.

---

## 3. The Training Pipeline (`training/` directory)

Here is the exact purpose of every file in the training directory:

* **`model.py`**: Defines the Keras (TensorFlow) model described above. It contains functions for the `se_block`, the `res_se_block` (which adds Batch Normalization and skip connections), and `build_model` which wires the Conv2D layers to the Dense layers.
* **`train.py`**: The main training loop. It processes the dataset, splits it into validation/training, enables Mixed Precision (using `float16` to train faster on the GPU), and uses the `Huber` loss function (which is robust to outliers compared to Mean Squared Error). Contains callbacks for early stopping and model checkpointing.
* **`dataset.py`**: Handles parsing the massive dataset (e.g., `chessData.csv`). It converts FENs (string representations of boards) into the numerical tensors ($8 \times 8 \times 12$ numpy arrays) expected by the model. It incorporates caching so you don't parse string data every epoch.
* **`config.py`**: A central location for hyperparameter variables like `FILTERS`, `NUM_SE_BLOCKS`, `LEARNING_RATE`, and dataset paths.
* **`export_onnx.py`**: Once TensorFlow finishes training to form `.h5` or `.keras` checkpoints, this script freezes the computation graph and exports it to `.onnx`. ONNX is a universal C++ model format; without it, embedding the massive Python/TensorFlow environment inside your high-performance C++ engine would be impossible.

---

## 4. The C++ Engine (`engine_cpp/` directory)

The C++ Engine is the performance-critical part of the project. It handles millions of operations per second to search for the best move.

### Engine Component Breakdown:
* **`bitboard.cpp` / `.h`**: Represents the chess board using 64-bit unsigned integers (`uint64_t`). Since a board has 64 squares, each bit corresponds to a square. This handles magical masks for sliding pieces (Rooks/Bishops) with extreme speed.
* **`position.cpp` / `.h`**: Maintains the current mathematical state of the game. It controls the FEN string, player turn, castling rights, and is the interface used to "make" and "undo" moves during search.
* **`movegen.cpp` / `.h`**: Generates all perfectly legal moves for a given position using bitwise operations, accounting for pins and checks.
* **`perft.cpp` / `.h`**: "Performance Test". Helps debug move generation by walking the tree of all possible moves to a certain depth and counting the total number of exact nodes, comparing it to known mathematical truths.
* **`search.cpp` / `.h`**: Algorithms (like standard Alpha-Beta pruning or Minimax) to search the tree of moves generated by `movegen` to find the move that maximizes the Neural Network's evaluation score.

---

## 5. Line-by-Line Code Analysis of the Engine Bridge

The engine requires bridge files that connect standard input/output, search, and neural networks together. Below is a line-by-line explanation of the most critical files.

### A. `src/main.cpp` (The Entry Point)
This file represents the starting point of the compiled program.

```cpp
#include "bitboard.h" // Includes bitboard engine definitions
#include "position.h" // Handles board state
#include "movegen.h"  // Generates legal chess moves
#include "perft.h"    // Debugging utility
#include "uci.h"      // Universal Chess Interface handler

#include <iostream>
// ... [other standard includes]

// 1. struct PerftTest definitions
// This struct defines known chess positions (FENs) and how many exact legal moves 
// exist at depths 1, 2, 3, etc. This is used strictly for debugging.
struct PerftTest {
    std::string fen;
    std::string name;
    std::vector<std::pair<int, uint64_t>> expected;
};

// 2. The Engine's Start Line
int main(int argc, char* argv[]) {

    // 3. CRITICAL: Initialize the Bitboards. This generates pre-calculated attack 
    // tables (like "where can a knight jump from square X") into memory so that
    // the engine doesn't have to calculate them on the fly.
    BB::init();

    // 4. Command line arguments check
    // If you run `./engine_cpp --perft` in the terminal, it triggers the debug suite.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--perft") == 0) {
            return run_perft_suite(); // Executes movegen debugging
        }
    }

    // 5. Default mode: UCI mode
    // If no flags are provided, we initialize the UCI interface. This is what lets
    // GUIs like Arena or Lichess talk to the engine.
    UCI uci;
    
    // 6. Infinitely loop waiting for commands from the user/GUI over Standard Input.
    uci.loop();

    return 0; // Exit successfully when the loop ends.
}
```

### B. `src/uci.cpp` (The Brain's Communication Layer)
The engine does not have a visual chessboard. It takes raw text commands via standard input and replies.

```cpp
// 1. Core loop handling input text line-by-line
void UCI::loop() {
    std::string line;
    // std::getline freezes the program until a line of text is piped in
    while (std::getline(std::cin, line)) { 
        // Trim whitespace and carriage returns from the end
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd; // Extract the first word of the command

        // Route the command correctly:
        if (cmd == "uci")             cmd_uci(); // GUI asks us to identify ourselves
        else if (cmd == "isready")    cmd_isready(); // GUI asks if we loaded our weights yet
        else if (cmd == "ucinewgame") cmd_ucinewgame(); // Clear board for a new game
        else if (cmd == "position")   cmd_position(line.substr(cmd.size())); // Set the pieces
        else if (cmd == "go")         cmd_go(line.substr(cmd.size())); // GUI tells us to START THINKING
        else if (cmd == "setoption")  cmd_setoption(line.substr(cmd.size())); // Config updates (e.g. model path)
        else if (cmd == "quit")       { cmd_quit(); return; } // Terminate engine
    }
}

// 2. When the GUI literally tells the engine "go wtime 300000 btime 300000"
void UCI::cmd_go(const std::string& args) {
    // Determine how much time we are allowed to use to think
    int time_ms = parse_time(args); 
    if (time_ms <= 0) time_ms = 2000; // Hard default to 2 seconds if none given

    // BLOCKING CALL: Submit the current position to the heavy search algorithm.
    // The search function uses Alpha-Beta pruning to find the best possible move, it
    // evaluates lines using the neural network until the time_ms limit runs out.
    Move best = search_.search(pos_, time_ms); 

    if (best != MOVE_NONE) {
        // Reply via standard output to the GUI mapping the move (e.g., "bestmove e2e4")
        std::cout << "bestmove " << best.to_string() << std::endl;
    } else {
        std::cout << "bestmove 0000" << std::endl; // No legal moves fallback
    }
}
```

### C. `src/nn_eval.cpp` (Where C++ Meets Deep Learning)
This file relies on the `onnxruntime_cxx_api` to run inference using the `.onnx` Neural Network model exported from Python.

```cpp
// 1. Load Model from Path
bool NNEvaluator::load_model(const string& model_path) {
    try {
        Ort::SessionOptions opts; // ONNX Runtime configuration
        opts.SetIntraOpNumThreads(1); // Force 1 thread so multiple search nodes don't clash threads
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load the .onnx file from disk into RAM (the 'Session')
        ort_state_->session = make_unique<Ort::Session>(
            ort_state_->env, model_path.c_str(), opts
        );

        // ... [caching memory input/output layer string names for faster pointer access later] ...

        loaded_ = true;
        return true;
    } catch (...) {
        return false; // Model didn't load properly
    }
}

// 2. Evaluate a specific chess position (This happens millions of times per game)
float NNEvaluator::evaluate(const Position& pos) {
    if (!loaded_) return 0.0f; // Safety fallback

    // Array allocations mimicking the Python structure exactly
    array<float, 8 * 8 * 12> board_data{}; 
    array<float, 8> meta_data{};
    
    // C++ function to scan the bitboards and convert the pieces into a format
    // the Neural Network accepts (the 8x8x12 float tensor).
    encode_board(pos, board_data.data()); 
    encode_metadata(pos, meta_data.data());

    // Define the dimensions of the tensors for ONNX Memory mapping
    array<int64_t, 4> board_shape = {1, 8, 8, 12}; // Number of boards(1), 8 rows, 8 cols, 12 channels
    array<int64_t, 2> meta_shape = {1, 8}; // 1 board, 8 metadata features

    // Wrap the raw C++ memory vectors into specific ONNX Value Tensors
    Ort::Value board_tensor = Ort::Value::CreateTensor<float>(
        ort_state_->memory_info, board_data.data(), board_data.size(),
        board_shape.data(), board_shape.size()
    );
    // ... [and we do the same for meta_tensor] ...

    vector<Ort::Value> inputs;
    inputs.emplace_back(move(board_tensor));
    inputs.emplace_back(move(meta_tensor));

    // EXECUTE INFERENCE: This forces the data through the convolutional and dense layers 
    // of the SE-ResNet loaded inside the ONNX Session.
    auto output_tensors = ort_state_->session->Run(
        Ort::RunOptions{nullptr},
        ort_state_->input_names.data(), inputs.data(), inputs.size(),
        ort_state_->output_names.data(), ort_state_->output_names.size()
    );

    // Extract the raw floating point value from the result (Our 'Eval' score)
    float nn_score = output_tensors[0].GetTensorData<float>()[0];
    
    // Fallback baseline: Evaluate raw piece material.
    // This helps the AI quickly realize that sacrificing a Queen is bad early in training,
    // smoothing out errors made solely by the neural network.
    float mat_score = material_eval(pos);
    
    // Return a composite evaluation: Neural Network Score heavily weighted against basic piece values
    return nn_score + (mat_score * 0.5f); 
}
```

## Summary Workflow: How a Move is Made
1. You run the compiled `engine_cpp` executable.
2. A GUI (like Arena) connects to it via Standard I/O pipelines and sends `uci`. The engine replies `uciok` and exposes its configuration to the UI.
3. The GUI sends `position startpos moves e2e4`, setting up the initial position and executing the `e2e4` move globally on the `position` object.
4. The GUI sends a `go wtime 30000 btime 30000` string, indicating it's your turn and you have 30 seconds left on your clock.
5. `uci.cpp` intercepts the `go` command, calculates the engine has approximately ~2 seconds per move to think, and fires `search_.search()`.
6. Inside `search.cpp`, it recursively calls `movegen.cpp` to map out the tree of all possible replies to `e2e4`.
7. At the leaves of every search branch, it formats the board specifically for NumPy shape matching, and pushes the board array through `nn_eval.cpp`.
8. The ONNX model (the SE-ResNet) spits out a floating-point number representing who is winning mathematically.
9. After 2 seconds, the Alpha-Beta pruning algorithm terminates, bubbling up the move node that resulted in the highest neural network score.
10. `uci.cpp` prints `bestmove e7e5` back to the console for the GUI to consume and animate visually.
