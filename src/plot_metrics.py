from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import RUNS_DIR, EXPERIMENT_NAME


RUN_DIR = RUNS_DIR / EXPERIMENT_NAME
EVAL_DIR = RUN_DIR / "evaluation"
PLOTS_DIR = EVAL_DIR / "plots"

RESULTS_CSV = RUN_DIR / "results.csv"
METRICS_CSV = EVAL_DIR / "metrics_test.csv"
REPORT_CSV = EVAL_DIR / "classification_report_test.csv"


def ensure_files_exist() -> None:
    required_files = [RESULTS_CSV, METRICS_CSV, REPORT_CSV]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(results_df: pd.DataFrame) -> None:
    epochs = results_df["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results_df["train/loss"], marker="o", label="Train Loss")
    plt.plot(epochs, results_df["val/loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results_df["metrics/accuracy_top1"], marker="o", label="Top-1 Accuracy")
    plt.plot(epochs, results_df["metrics/accuracy_top5"], marker="o", label="Top-5 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curves.png", dpi=300)
    plt.close()


def plot_main_metrics(metrics_df: pd.DataFrame) -> None:
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "top1_accuracy",
        "top3_accuracy",
        "macro_auc",
        "weighted_auc",
        "micro_auc",
    ]

    available_metrics = [name for name in metric_names if name in metrics_df.columns]
    values = [float(metrics_df.iloc[0][name]) for name in available_metrics]

    plt.figure(figsize=(13, 6))
    plt.bar(available_metrics, values)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Main Classification Metrics")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "main_metrics_bar.png", dpi=300)
    plt.close()


def plot_per_class_f1(report_df: pd.DataFrame) -> None:
    excluded_rows = {"accuracy", "macro avg", "weighted avg"}
    class_report = report_df[~report_df.index.isin(excluded_rows)].copy()

    if "f1-score" not in class_report.columns:
        raise ValueError("Column 'f1-score' not found in classification_report_test.csv")

    class_report = class_report[class_report["support"] > 0].copy()
    class_report = class_report.sort_values(by="f1-score", ascending=True)

    plt.figure(figsize=(12, 10))
    plt.barh(class_report.index, class_report["f1-score"])
    plt.xlim(0.0, 1.05)
    plt.xlabel("F1-score")
    plt.ylabel("Class")
    plt.title("Per-Class F1-score")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_class_f1.png", dpi=300)
    plt.close()


def plot_per_class_precision_recall(report_df: pd.DataFrame) -> None:
    excluded_rows = {"accuracy", "macro avg", "weighted avg"}
    class_report = report_df[~report_df.index.isin(excluded_rows)].copy()
    class_report = class_report[class_report["support"] > 0].copy()
    class_report = class_report.sort_values(by="support", ascending=False)

    top_n = min(15, len(class_report))
    class_report = class_report.head(top_n)

    x = range(len(class_report))

    plt.figure(figsize=(12, 6))
    plt.plot(x, class_report["precision"], marker="o", label="Precision")
    plt.plot(x, class_report["recall"], marker="o", label="Recall")
    plt.plot(x, class_report["f1-score"], marker="o", label="F1-score")
    plt.xticks(x, class_report.index, rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Metric value")
    plt.title("Precision / Recall / F1-score for top-supported test classes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_class_precision_recall_f1.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_files_exist()

    results_df = pd.read_csv(RESULTS_CSV)
    metrics_df = pd.read_csv(METRICS_CSV)
    report_df = pd.read_csv(REPORT_CSV, index_col=0)

    plot_training_curves(results_df)
    plot_main_metrics(metrics_df)
    plot_per_class_f1(report_df)
    plot_per_class_precision_recall(report_df)

    print("Plots created successfully.")
    print(f"Saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()