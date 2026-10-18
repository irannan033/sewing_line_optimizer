"""Central configuration: paths, seeds and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "garments_worker_productivity.csv"  # UCI dataset (optional)
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SEED = 42
SHIFT_MINUTES = 480  # 8 hour shift

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, RAW_CSV.parent):
    _d.mkdir(parents=True, exist_ok=True)
