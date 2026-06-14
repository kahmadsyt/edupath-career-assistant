from pathlib import Path

# Root project directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Main folders
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_DIR = ROOT_DIR / "outputs"
CONFIG_DIR = ROOT_DIR / "config"

CONFIG_FILE = CONFIG_DIR / "config.yaml"