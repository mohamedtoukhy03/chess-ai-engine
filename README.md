# Chess AI Engine

A neural network-powered chess engine built for a university NN course.

## Project Structure

```
chess-engine/
├── engine/                 # C++ UCI chess engine
│   ├── CMakeLists.txt
│   ├── types.h             # Core types & enums
│   ├── bitboard.h/cpp      # Bitboard utilities & attack tables
│   ├── move.h              # 16-bit move encoding
│   ├── position.h/cpp      # Board state & FEN parsing
│   ├── movegen.h/cpp       # Legal move generation
│   ├── nn_eval.h/cpp       # Neural network inference (ONNX)
│   ├── search.h/cpp        # Monte Carlo Tree Search
│   ├── uci.h/cpp           # UCI protocol
│   ├── perft.h/cpp         # Move generation validation
│   └── main.cpp            # Entry point
├── training/               # Python/PyTorch model training
│   ├── config.py           # Hyperparameters & paths
│   ├── model.py            # SE-ResNet CNN architecture
│   ├── dataset.py          # PGN/Stockfish → training data
│   ├── train.py            # Training loop
│   └── export_onnx.py      # PyTorch → ONNX export
├── server/                 # Python/FastAPI backend
│   └── app.py              # REST API + UCI engine manager
├── frontend/               # Web UI
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                 # Trained model files
└── data/pgn/               # PGN training data
```

## Quick Start

### 1. Build the C++ Engine
```bash
sudo apt install cmake g++ make
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Validate move generation
./chess_engine --perft
```

### 2. Train the Neural Network
```bash
cd training
pip install -r requirements.txt

# Generate test data (or place .pgn files in data/pgn/)
python dataset.py --synthetic
python train.py
python export_onnx.py
```

### 3. Run the Server
```bash
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Play
Open `http://localhost:8000` in your browser.

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Engine    | C++20     | Bitboard move generation + MCTS search |
| Model     | PyTorch   | SE-ResNet CNN for position evaluation |
| Inference | ONNX Runtime | Fast NN inference inside C++ engine |
| Backend   | FastAPI   | UCI engine wrapper + REST API |
| Frontend  | HTML/JS   | chessboard.js interactive UI |
