import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.utils.metrics import (
    build_classification_report,
    build_confusion_matrix_df,
    build_confusion_matrix_normalized_df,
    compute_classification_metrics,
    compute_multiclass_roc_auc,
    save_classification_report,
    save_confusion_matrix_df,
    save_metrics,
    save_roc_auc_report,
)
from src.utils.config import (
    MODELS_DIR,
    RUNS_DIR,
    EXPERIMENT_NAME,
    VAL_DIR,
    TEST_DIR,
    IMAGE_EXTENSIONS,
    DEVICE,
    TOP_K,
)


BEST_MODEL_PATH = RUNS_DIR / EXPERIMENT_NAME / "weights" / "best.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
OUTPUT_DIR = RUNS_DIR / EXPERIMENT_NAME / "evaluation"

EVAL_SPLIT = "test"


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Invalid class names file format.")

    return class_names


def get_split_dir() -> Path:
    if EVAL_SPLIT == "val":
        return VAL_DIR
    if EVAL_SPLIT == "test":
        return TEST_DIR
    raise ValueError("EVAL_SPLIT must be either 'val' or 'test'")


def collect_samples(split_dir: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []

    for class_id, class_name in enumerate(class_names):
        class_dir = split_dir / class_name

        if not class_dir.exists():
            continue

        if not class_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_id))

    return samples


def plot_roc_curves(
        roc_df: pd.DataFrame,
        curve_data: dict[str, np.ndarray],
        output_path: Path,
        hardest_classes_to_plot: int = 5,
) -> None:
    plt.figure(figsize=(11, 8))

    plt.plot(
        curve_data["micro_fpr"],
        curve_data["micro_tpr"],
        linewidth=2.5,
        label="Micro-average ROC",
    )
    plt.plot(
        curve_data["macro_fpr"],
        curve_data["macro_tpr"],
        linewidth=2.5,
        label="Macro-average ROC",
    )

    hardest_df = roc_df.sort_values(by="auc", ascending=True).head(hardest_classes_to_plot)

    for _, row in hardest_df.iterrows():
        fpr = np.array(ast.literal_eval(row["fpr"]) if isinstance(row["fpr"], str) else row["fpr"])
        tpr = np.array(ast.literal_eval(row["tpr"]) if isinstance(row["tpr"], str) else row["tpr"])
        class_name = row["class_name"]
        class_auc = float(row["auc"])
        support = int(row["support"])

        plt.plot(
            fpr,
            tpr,
            linestyle="--",
            linewidth=1.5,
            label=f"{class_name} (AUC={class_auc:.4f}, n={support})",
        )

    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=2, label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves on {EVAL_SPLIT.upper()} split")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(16, 13))
    plt.imshow(cm_df.values, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=90)
    plt.yticks(range(len(cm_df.index)), cm_df.index)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_model() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(CLASS_NAMES_PATH)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")

    split_dir = get_split_dir()
    print(f"Evaluation split: {EVAL_SPLIT}")
    print(f"Using directory: {split_dir}")
    print(f"Device: {DEVICE}")

    model = YOLO(str(BEST_MODEL_PATH))
    samples = collect_samples(split_dir, class_names)

    if not samples:
        raise ValueError(
            f"No valid samples found in {split_dir}. "
            "Split directory must contain class folders with images."
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    top_k_predictions: list[list[int]] = []
    y_score: list[list[float]] = []

    print(f"Found {len(samples)} images.")
    print("Starting evaluation...")

    for i, (image_path, true_class_id) in enumerate(samples, start=1):
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

        # На випадок чисельних артефактів
        prob_sum = prob_vector.sum()
        if prob_sum <= 0:
            raise ValueError(f"Invalid probability vector for image: {image_path}")
        prob_vector = prob_vector / prob_sum

        pred_class_id = int(np.argmax(prob_vector))

        sorted_indices = np.argsort(prob_vector)[::-1]
        # Keep enough predictions for top-3 reporting while respecting configured TOP_K.
        top_indices = sorted_indices[:max(TOP_K, 3)].tolist()

        y_true.append(true_class_id)
        y_pred.append(pred_class_id)
        top_k_predictions.append(top_indices)
        y_score.append(prob_vector.tolist())

        if i % 500 == 0:
            print(f"Processed {i}/{len(samples)} images...")

    y_score_np = np.array(y_score, dtype=float)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        top_k_predictions=top_k_predictions,
    )

    report_df = build_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    cm_df = build_confusion_matrix_df(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    cm_norm_df = build_confusion_matrix_normalized_df(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    roc_df, auc_summary, curve_data = compute_multiclass_roc_auc(
        y_true=y_true,
        y_score=y_score_np,
        class_names=class_names,
    )

    metrics.update(auc_summary)
    metrics["evaluation_split"] = EVAL_SPLIT
    metrics["num_samples"] = len(samples)
    metrics["num_present_classes"] = len(set(y_true))

    metrics_path = OUTPUT_DIR / f"metrics_{EVAL_SPLIT}.csv"
    report_path = OUTPUT_DIR / f"classification_report_{EVAL_SPLIT}.csv"
    roc_auc_path = OUTPUT_DIR / f"roc_auc_summary_{EVAL_SPLIT}.csv"
    roc_plot_path = OUTPUT_DIR / f"roc_curves_{EVAL_SPLIT}.png"

    cm_csv_path = OUTPUT_DIR / f"confusion_matrix_{EVAL_SPLIT}.csv"
    cm_norm_csv_path = OUTPUT_DIR / f"confusion_matrix_normalized_{EVAL_SPLIT}.csv"
    cm_plot_path = OUTPUT_DIR / f"confusion_matrix_{EVAL_SPLIT}.png"
    cm_norm_plot_path = OUTPUT_DIR / f"confusion_matrix_normalized_{EVAL_SPLIT}.png"

    save_metrics(metrics, metrics_path)
    save_classification_report(report_df, report_path)
    save_roc_auc_report(roc_df, roc_auc_path)
    save_confusion_matrix_df(cm_df, cm_csv_path)
    save_confusion_matrix_df(cm_norm_df, cm_norm_csv_path)

    plot_roc_curves(roc_df, curve_data, roc_plot_path)
    plot_confusion_matrix(cm_df, cm_plot_path, f"Confusion Matrix on {EVAL_SPLIT.upper()} split")
    plot_confusion_matrix(
        cm_norm_df,
        cm_norm_plot_path,
        f"Normalized Confusion Matrix on {EVAL_SPLIT.upper()} split",
    )

    print("\nEvaluation completed successfully.")
    print("\nMain metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Classification report saved to: {report_path}")
    print(f"ROC/AUC summary saved to: {roc_auc_path}")
    print(f"ROC plot saved to: {roc_plot_path}")
    print(f"Confusion matrix CSV saved to: {cm_csv_path}")
    print(f"Normalized confusion matrix CSV saved to: {cm_norm_csv_path}")
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    print(f"Normalized confusion matrix plot saved to: {cm_norm_plot_path}")


if __name__ == "__main__":
    evaluate_model()
