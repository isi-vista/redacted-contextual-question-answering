"""Paths to common directories and files."""

from pathlib import Path

# High-level directories
TOP_LEVEL_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = TOP_LEVEL_DIR / ".cache"
DATA_DIR = TOP_LEVEL_DIR / "data"
EXAMPLE_DIR = TOP_LEVEL_DIR / "examples"

# Cache directories
TRANSFORMERS_CACHE_DIR = CACHE_DIR / "transformers_cache"

# Data directories
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = DATA_DIR / "models"
RCQA_EVAL_DIR = DATA_DIR / "rcqa_eval"
SLURM_LOG_DIR = DATA_DIR / "slurm_logs"

# Automatically create most directories
dirs = [
    CACHE_DIR,
    DATA_DIR,
    EVAL_DIR,
    MODELS_DIR,
    RCQA_EVAL_DIR,
    SLURM_LOG_DIR,
    TRANSFORMERS_CACHE_DIR,
]
for directory in dirs:
    directory.mkdir(parents=True, exist_ok=True)

# Other directories and files
DOTENV_PATH = TOP_LEVEL_DIR / ".env"
