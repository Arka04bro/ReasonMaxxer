from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("REASONMAXXER_DATA_DIR", PROJECT_ROOT / "data"))
BENCHMARK_DIR = DATA_DIR / "benchmarks"

MATH500_DATA_FILE = DATA_DIR / "math500_cached.json"
GSM8K_DATA_FILE = DATA_DIR / "gsm8k_cached.json"

# Local benchmark JSON files are optional. If they do not exist, pass --records_file.
BENCHMARK_DATA_FILES = {
    "aime24": BENCHMARK_DIR / "aime24.json",
    "amc23": BENCHMARK_DIR / "amc23.json",
    "minerva_math": BENCHMARK_DIR / "minerva_math.json",
    "olympiadbench": BENCHMARK_DIR / "olympiadbench.json",
}

DATASET_CONFIG = {
    "math500": {
        "hf_name": "nlile/hendrycks-MATH-benchmark",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "solution",
    },
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "config_name": "main",
        "split": "test",
        "problem_key": "question",
        "answer_key": "answer",
    },
}

TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 8192
MAX_MODEL_LEN = 8192
SEED = 42
TENSOR_PARALLEL_SIZE = 1
BATCH_SIZE = 8
STOP_SEQUENCES = [
    "Human:",
    "\nHuman:",
    "\n\nHuman:",
    "Assistant:",
    "\nAssistant:",
    "\n\nAssistant:",
]
