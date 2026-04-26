# Chess Engine Deployment Pipeline

This repo already had most of the requested stack implemented:
- custom C++ board representation + legal move generation
- alpha-beta search
- ONNX Runtime evaluator integration
- UCI loop

The files below were added/updated to complete the requested deployment pipeline.

## Added Files

- `colab/convert_model.py`  
  Keras (`.keras`) -> ONNX converter using `tf2onnx` with dual input signatures:
  - `board`: `(None, 8, 8, 12)`
  - `meta`: `(None, 8)`

- `engine/include/engine_api.h`  
  Public deployment API:
  - `std::string get_best_move(const std::string& fen, int depth, const std::string& model_path);`

- `engine/src/engine_api.cpp`  
  Implements `get_best_move(...)` by:
  - initializing bitboards
  - loading FEN into `Position`
  - loading ONNX model via `NNEvaluator`
  - running fixed-depth alpha-beta via `AlphaBetaSearch::get_best_move`

- `engine/src/deploy_main.cpp`  
  One-shot CLI for deployment:
  - `--fen "<fen>"`
  - `--depth <n>`
  - `--model <path.onnx>`

- `Dockerfile`  
  Builds and packages the engine with ONNX Runtime CPU.

## Updated Files

- `engine/src/nn_eval.cpp`, `engine/include/nn_eval.h`
  - Added metadata encoder for 8 features:
    `[turn, wK, wQ, bK, bQ, ep_present, ep_file_norm, in_check]`
  - Added perspective normalization for black-to-move:
    - rotate board 180
    - swap colors
  - Updated board encoding to flattened NHWC `(8, 8, 12)` for Keras-style ONNX
  - Added support for both:
    - dual-input ONNX models (`board`, `meta`)
    - legacy single-input models (board-only)
  - Input/output names are queried from the ONNX session instead of hardcoding.

- `engine/include/search.h`, `engine/src/search.cpp`
  - Added fixed-depth entry point:
    - `Move get_best_move(Position& pos, int depth)`

- `engine/CMakeLists.txt`
  - Included new source files
  - Added `chess_engine_cli` target

## Project Structure (deployment-relevant)

```text
.
├── colab/
│   ├── NN.ipynb
│   ├── chessData.csv
│   ├── chess_se_resnet_best.keras
│   └── convert_model.py
├── engine/
│   ├── include/
│   │   ├── engine_api.h
│   │   ├── nn_eval.h
│   │   ├── position.h
│   │   └── search.h
│   ├── src/
│   │   ├── deploy_main.cpp
│   │   ├── engine_api.cpp
│   │   ├── nn_eval.cpp
│   │   ├── position.cpp
│   │   ├── movegen.cpp
│   │   └── search.cpp
│   └── CMakeLists.txt
├── CMakeLists.txt
└── Dockerfile
```

## Build and Run

### 1) Export ONNX model

```bash
python3 colab/convert_model.py \
  --keras "colab/chess_se_resnet_best.keras" \
  --onnx "models/chess_eval.onnx"
```

### 2) Native build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 3) Run one-shot best move search

```bash
./build/engine/chess_engine_cli \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --depth 4 \
  --model models/chess_eval.onnx
```

### 4) Run UCI engine

```bash
./build/engine/chess_engine
```

Then in UCI:
```text
uci
setoption name ModelPath value models/chess_eval.onnx
isready
position startpos
go wtime 60000 btime 60000
```

## Docker

### Build image

```bash
docker build -t chess-engine-onnx .
```

### Run one-shot search

```bash
docker run --rm chess-engine-onnx \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --depth 4 \
  --model /app/models/chess_eval.onnx
```
