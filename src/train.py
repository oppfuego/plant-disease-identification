from ultralytics import YOLO

from src.utils.config import (
    DATASET_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RUNS_DIR,
    EXPERIMENT_NAME,
    YOLO_MODEL_WEIGHTS,
    DEVICE,
    SEED,
)


def main() -> None:
    print("Starting training...")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model weights: {YOLO_MODEL_WEIGHTS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")

    model = YOLO(YOLO_MODEL_WEIGHTS)

    model.train(
        data=str(DATASET_DIR),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        project=str(RUNS_DIR),
        name=EXPERIMENT_NAME,
        verbose=True,
        device=DEVICE,
        cache=True,
        workers=2,
        seed=SEED,
        pretrained=True,
        val=True,
    )

    print("\nTraining completed successfully.")
    print(f"Results directory: {RUNS_DIR / EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()