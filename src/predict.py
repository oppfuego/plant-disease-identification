import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.utils.config import MODELS_DIR, RUNS_DIR, EXPERIMENT_NAME, TOP_K, DEVICE


BEST_MODEL_PATH = RUNS_DIR / EXPERIMENT_NAME / "weights" / "best.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


def predict_image(
        model: YOLO,
        image_path: Path,
        class_names: list[str],
        top_k: int = TOP_K,
) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(
        source=str(image_path),
        verbose=False,
        device=DEVICE,
    )

    result = results[0]
    probs = result.probs

    if probs is None or probs.data is None:
        raise ValueError(f"No probabilities returned for image: {image_path}")

    prob_vector = probs.data.detach().cpu().numpy().astype(float)
    prob_vector = prob_vector / prob_vector.sum()

    sorted_indices = np.argsort(prob_vector)[::-1]
    top_indices = sorted_indices[:top_k]

    top_predictions = []
    for idx in top_indices:
        idx = int(idx)
        conf = float(prob_vector[idx])
        top_predictions.append(
            {
                "class_id": idx,
                "class_name": class_names[idx],
                "confidence": conf,
            }
        )

    top1_index = int(top_indices[0])
    top1_conf = float(prob_vector[top1_index])

    return {
        "image": str(image_path),
        "predicted_class_id": top1_index,
        "predicted_class_name": class_names[top1_index],
        "confidence": top1_conf,
        "top_predictions": top_predictions,
    }


def print_prediction(prediction: dict) -> None:
    print(f"\nImage: {prediction['image']}")
    print(f"Predicted class: {prediction['predicted_class_name']}")
    print(f"Confidence: {prediction['confidence']:.4f}")

    print("\nTop predictions:")
    for i, pred in enumerate(prediction["top_predictions"], start=1):
        print(
            f"{i}. {pred['class_name']} "
            f"(class_id={pred['class_id']}, confidence={pred['confidence']:.4f})"
        )


def main() -> None:
    class_names = load_class_names(CLASS_NAMES_PATH)
    model = load_model(BEST_MODEL_PATH)

    image_input = input("Enter image path: ").strip()
    image_path = Path(image_input)

    prediction = predict_image(
        model=model,
        image_path=image_path,
        class_names=class_names,
        top_k=TOP_K,
    )

    print_prediction(prediction)


if __name__ == "__main__":
    main()