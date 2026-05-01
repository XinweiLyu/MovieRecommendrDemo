import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

scripts = [
    "01_data_cleaning.py",
    "02_eda_visualization.py",
    "03_modeling.py",
    "04_recommendation_demo.py",
]

for script in scripts:
    print("\n" + "=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    subprocess.run([sys.executable, str(SRC_DIR / script)], check=True)

print("\nAll steps completed successfully.")
