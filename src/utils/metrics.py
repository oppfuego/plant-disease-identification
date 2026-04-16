from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_top_k_accuracy(
        y_true: list[int],
        top_k_predictions: list[list[int]],
        k: int,
) -> float:
    if not y_true:
        return 0.0

    correct = 0
    for true_label, pred_indices in zip(y_true, top_k_predictions):
        if true_label in pred_indices[:k]:
            correct += 1

    return correct / len(y_true)


def compute_classification_metrics(
        y_true: list[int],
        y_pred: list[int],
        top_k_predictions: list[list[int]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if top_k_predictions is not None:
        metrics["top1_accuracy"] = compute_top_k_accuracy(y_true, top_k_predictions, k=1)
        metrics["top3_accuracy"] = compute_top_k_accuracy(y_true, top_k_predictions, k=3)

    return metrics


def build_classification_report(
        y_true: list[int],
        y_pred: list[int],
        class_names: list[str],
) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def build_confusion_matrix_df(
        y_true: list[int],
        y_pred: list[int],
        class_names: list[str],
) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


def build_confusion_matrix_normalized_df(
        y_true: list[int],
        y_pred: list[int],
        class_names: list[str],
) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    return pd.DataFrame(cm_norm, index=class_names, columns=class_names)


def compute_multiclass_roc_auc(
        y_true: list[int],
        y_score: np.ndarray,
        class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    if len(y_true) == 0:
        raise ValueError("y_true is empty.")

    if y_score.ndim != 2:
        raise ValueError("y_score must be a 2D array: [num_samples, num_classes].")

    if len(y_true) != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same number of samples.")

    if y_score.shape[1] != len(class_names):
        raise ValueError("y_score second dimension must match number of classes.")

    row_sums = y_score.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_score = y_score / row_sums

    classes = list(range(len(class_names)))
    y_true_bin = label_binarize(y_true, classes=classes)

    roc_rows = []
    auc_values = []
    supports = []
    per_class_fpr: dict[int, np.ndarray] = {}
    per_class_tpr: dict[int, np.ndarray] = {}

    for class_id, class_name in enumerate(class_names):
        y_true_class = y_true_bin[:, class_id]
        y_score_class = y_score[:, class_id]

        positive_count = int(np.sum(y_true_class))
        negative_count = int(len(y_true_class) - positive_count)

        if positive_count == 0 or negative_count == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true_class, y_score_class)
        class_auc = auc(fpr, tpr)

        auc_values.append(float(class_auc))
        supports.append(positive_count)
        per_class_fpr[class_id] = fpr
        per_class_tpr[class_id] = tpr

        roc_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "auc": float(class_auc),
                "support": positive_count,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }
        )

    roc_df = pd.DataFrame(roc_rows)
    if roc_df.empty:
        raise ValueError("ROC/AUC cannot be computed: no valid classes found in evaluation split.")

    macro_auc = float(np.mean(auc_values))
    weighted_auc = float(np.average(auc_values, weights=supports))

    micro_fpr, micro_tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    micro_auc = float(auc(micro_fpr, micro_tpr))

    all_fpr = np.unique(np.concatenate(list(per_class_fpr.values())))
    mean_tpr = np.zeros_like(all_fpr)

    for class_id in per_class_fpr:
        mean_tpr += np.interp(all_fpr, per_class_fpr[class_id], per_class_tpr[class_id])

    mean_tpr /= len(per_class_fpr)
    macro_curve_auc = float(auc(all_fpr, mean_tpr))

    summary = {
        "roc_num_present_classes": len(roc_df),
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "micro_auc": micro_auc,
        "macro_curve_auc": macro_curve_auc,
        "min_auc": float(np.min(auc_values)),
        "max_auc": float(np.max(auc_values)),
    }

    curve_data = {
        "micro_fpr": micro_fpr,
        "micro_tpr": micro_tpr,
        "macro_fpr": all_fpr,
        "macro_tpr": mean_tpr,
    }

    return roc_df, summary, curve_data


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False, encoding="utf-8")


def save_classification_report(report_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=True, encoding="utf-8")


def save_roc_auc_report(roc_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roc_df_to_save = roc_df.copy()
    roc_df_to_save["fpr"] = roc_df_to_save["fpr"].apply(str)
    roc_df_to_save["tpr"] = roc_df_to_save["tpr"].apply(str)
    roc_df_to_save.to_csv(output_path, index=False, encoding="utf-8")


def save_confusion_matrix_df(cm_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(output_path, index=True, encoding="utf-8")