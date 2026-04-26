# Chess AI Project Analysis (University Assignment Feasibility)

This project is a **chess position evaluation CNN** (not a typical image classification CV pipeline), and based on the code it is trainable from scratch, but the main time sink is dataset generation.

## Core Architecture

The network is a compact **SE-ResNet-style value network**:

- Input: `(12, 8, 8)` one-hot board planes (piece-type x color)
- Stem: `Conv3x3(12->128) + BN + ReLU`
- Trunk: `6 x` SE-Residual blocks (each has two `3x3` convs + BN + squeeze-excitation channel attention + skip connection)
- Head: `1x1 conv -> BN -> ReLU -> flatten -> FC(64->256) -> Dropout(0.3) -> FC(256->1) -> Tanh`

```python
# from training/model.py
self.input_conv = nn.Sequential(
    nn.Conv2d(INPUT_CHANNELS, NUM_FILTERS, 3, padding=1, bias=False),
    nn.BatchNorm2d(NUM_FILTERS),
    nn.ReLU(inplace=True),
)

self.res_tower = nn.Sequential(
    *[SEResBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)]
)

self.value_head = nn.Sequential(
    nn.Conv2d(NUM_FILTERS, 1, 1, bias=False),
    nn.BatchNorm2d(1),
    nn.ReLU(inplace=True),
    nn.Flatten(),
    nn.Linear(8 * 8, FC_HIDDEN),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(FC_HIDDEN, 1),
    nn.Tanh(),
)
```

Config confirms size/depth:

```python
# from training/config.py
NUM_RESIDUAL_BLOCKS = 6
NUM_FILTERS = 128
FC_HIDDEN = 256

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
```

Approximate trainable parameters are around **1.85M** (moderate size).

## Pre-trained vs Scratch

This is set up for **from-scratch training**. There is no ImageNet/backbone download and no pretrained checkpoint dependency in training startup.

- `train.py` initializes `ChessEvalNet()` fresh.
- It loads data from `dataset.pt` or builds it from PGNs/Stockfish labels.
- `export_onnx.py` can load a saved local checkpoint, but if missing it exports an untrained model.

```python
# from training/train.py
if os.path.exists(DATASET_PATH):
    tensors, labels = load_dataset()
else:
    tensors, labels = build_dataset()
```

## Academic Complexity

For a university assignment, this is **moderately complex**, not trivial:

- **Advanced elements present**: residual connections, SE attention blocks, AdamW optimizer, cosine LR schedule, early stopping, custom metric (sign accuracy).
- **Missing high-complexity elements**: no novel/custom loss, no multi-head architecture (policy+value), no transformer blocks, no self-play RL loop.

Verdict: likely acceptable as “complex” for many undergrad courses, especially if you explain the mathematics of residual/SE blocks and label generation. For a very demanding graduate-level “complex task,” it may feel somewhat standard.

## Training Feasibility (Single GPU, 1-2 Days)

- **Model training itself**: feasible on one GPU in <=1-2 days (often much faster, usually hours), because input is tiny (`8x8`) and model is ~1.85M params.
- **Main bottleneck is data generation**, not GPU training.
  - Default config requests `NUM_POSITIONS = 400000` with Stockfish depth 12.
  - Stockfish evaluation is CPU-bound and can take many hours or even days depending on CPU/PGN quality.

```python
# from training/config.py
STOCKFISH_PATH = "/usr/bin/stockfish"
STOCKFISH_DEPTH = 12
NUM_POSITIONS = 400000
```

## Potential Roadblocks / Time Sinks

- **Dataset labeling runtime**: `engine.analyse(... depth=12)` per position is expensive.
- **Data availability**: if no PGNs exist in `data/pgn`, pipeline falls back to synthetic random positions (fast, but low-quality signal).
- **Stockfish setup**: hardcoded path `/usr/bin/stockfish`; if unavailable, behavior changes.
- **Memory/disk usage**: storing and loading all tensors/labels for 400k positions can become heavy.
- **Model quality risk**: PGN-result labels (fallback mode) are noisier than Stockfish-evaluated labels.

