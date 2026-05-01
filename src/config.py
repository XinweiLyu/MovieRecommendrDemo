from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
RECOMMENDATION_DIR = PROJECT_ROOT / "outputs" / "recommendations"

RANDOM_STATE = 42
RATING_MIN = 0.5
RATING_MAX = 5.0

for path in [PROCESSED_DATA_DIR, FIGURE_DIR, MODEL_DIR, RECOMMENDATION_DIR]:
    path.mkdir(parents=True, exist_ok=True)
