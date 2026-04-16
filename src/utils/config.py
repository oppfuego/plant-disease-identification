from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

SOURCE_DIR = RAW_DATA_DIR / "color"
DATASET_DIR = RAW_DATA_DIR / "plant-dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"

NUM_CLASSES = 38
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42
TOP_K = 3

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

YOLO_MODEL_WEIGHTS = "yolo11n-cls.pt"
EXPERIMENT_NAME = "plant_disease_cls"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# source .venv/bin/activate
# python -m src.prepare_data
# python -m src.train
# python -m src.evaluate