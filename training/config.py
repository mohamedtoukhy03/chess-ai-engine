from pathlib import Path

# Project root (repository)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset source
CSV_PATH = str(PROJECT_ROOT / "data" / "chessData.csv")
N_ROWS = 500_000
COL_FEN = "FEN"
COL_EVAL = "Evaluation"

# Model/training hyperparameters
BATCH_SIZE = 4096
EPOCHS = 50
LEARNING_RATE = 3e-4
FILTERS = 128
NUM_SE_BLOCKS = 9
META_DIM = 8
VAL_SPLIT = 0.02
RANDOM_SEED = 42

# Evaluation normalization
EVAL_SCALE_CP = 400.0
CLIP_CP = 10_000.0

# Optional metadata enrichment
IS_CHECK_WORKERS = 8  # 0 disables python-chess in-check pass
USE_MIXED_PRECISION = True

# Artifacts
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_PATH = str(MODELS_DIR / "chess_se_resnet_best.keras")
DATASET_CACHE_PATH = str(DATA_DIR / "dataset_tf.npz")
ONNX_EXPORT_PATH = str(MODELS_DIR / "chess_eval.onnx")
