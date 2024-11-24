import os

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(ROOT_DIR, "../data/raw/finalDs.csv")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "../data/processed/")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_dataset.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "../outputs/")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models/")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots/")

# Ensure directories exist
for directory in [PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    if not os.path.exists(directory):
        print(f"[INFO] Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

# Columns
NUMERIC_COLUMNS = ["f1", "precision", "recall", "accuracy", "training_time", "inference_time"]
TARGET_COLUMN = "composite_score"