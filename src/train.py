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
    print("Starting merged-dataset training...")
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
        device=DEVICE,
        seed=SEED,
        pretrained=True,
        val=True,
        verbose=True,

        cache=True,
        workers=4,

        optimizer="AdamW",
        cos_lr=True,
        patience=8,
        weight_decay=0.0005,
        dropout=0.05,

        degrees=8.0,
        translate=0.05,
        scale=0.20,
        shear=1.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.20,

        erasing=0.10,

        mixup=0.0,
        cutmix=0.0,
    )

    print("\nTraining completed successfully.")
    print(f"Results directory: {RUNS_DIR / EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
