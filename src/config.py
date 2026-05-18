from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ExamCheatingDataset"
TRAIN_DIR = DATA_DIR / "train"          # the labelled images
RESULTS_DIR = PROJECT_ROOT / "results"  # comparison table (CSV)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"  # saved plots


RANDOM_SEED = 42
IMAGE_SIZE = (32, 32)

# --- Train / test split ---------------------------------------------------
TEST_SIZE = 0.20  # 80% train, 20% test
CATEGORY_TO_BINARY = {
    "normal act": 0,
    "looking friend": 1,
    "giving code": 1,
    "giving object": 1,
    "cheating": 1,
}
LABEL_NAMES = {0: "Normal", 1: "Fraud"}

# Image file types to accept (the dataset mixes .jpg / .JPG / .png / .PNG).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}