import json
import random
import shutil
from pathlib import Path

from src.utils.config import (
    SOURCE_DIR,
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    IMAGE_EXTENSIONS,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def validate_directory(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")


def reset_output_directories() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def get_class_names(source_dir: Path) -> list[str]:
    return sorted(folder.name for folder in source_dir.iterdir() if folder.is_dir())


def get_images_in_class(class_dir: Path) -> list[Path]:
    return sorted(
        file
        for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    files = files[:]
    random.shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    if len(train_files) + len(val_files) + len(test_files) != total:
        raise ValueError("Split sizes do not sum to total number of files.")

    return train_files, val_files, test_files


def copy_files(files: list[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination_dir / file_path.name)


def save_class_names(class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)


def count_images_in_class(class_dir: Path) -> int:
    return sum(
        1
        for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images_in_split(split_dir: Path) -> dict[str, int]:
    class_counts: dict[str, int] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            class_counts[class_dir.name] = count_images_in_class(class_dir)
    return class_counts


def check_same_classes(
        train_classes: list[str],
        val_classes: list[str],
        test_classes: list[str],
) -> None:
    train_set = set(train_classes)
    val_set = set(val_classes)
    test_set = set(test_classes)

    if train_set != val_set or train_set != test_set:
        print("\nMismatch between dataset splits:")
        print(f"Missing in val: {sorted(train_set - val_set)}")
        print(f"Extra in val: {sorted(val_set - train_set)}")
        print(f"Missing in test: {sorted(train_set - test_set)}")
        print(f"Extra in test: {sorted(test_set - train_set)}")
        raise ValueError("Class folders in train/val/test do not match.")


def print_split_stats(split_name: str, split_counts: dict[str, int]) -> None:
    total_images = sum(split_counts.values())
    print(f"\n{split_name} split:")
    print(f"  Number of classes: {len(split_counts)}")
    print(f"  Total images: {total_images}")

    for class_name, count in split_counts.items():
        print(f"  {class_name}: {count}")


def prepare_split() -> None:
    validate_directory(SOURCE_DIR, "Source dataset")

    class_names = get_class_names(SOURCE_DIR)

    if not class_names:
        raise ValueError(f"No class folders found in source directory: {SOURCE_DIR}")

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, but found {len(class_names)} in {SOURCE_DIR}"
        )

    reset_output_directories()

    print(f"Source directory: {SOURCE_DIR}")
    print(f"Found classes: {len(class_names)}")
    print(
        f"Split ratios -> train: {TRAIN_RATIO:.0%}, "
        f"val: {VAL_RATIO:.0%}, test: {TEST_RATIO:.0%}"
    )

    for class_name in class_names:
        class_dir = SOURCE_DIR / class_name
        image_files = get_images_in_class(class_dir)

        if not image_files:
            print(f"Skipping empty class: {class_name}")
            continue

        train_files, val_files, test_files = split_files(image_files)

        if len(train_files) == 0:
            raise ValueError(
                f"Class '{class_name}' has too few images for train split: {len(image_files)}"
            )
        if len(val_files) == 0:
            print(f"Warning: class '{class_name}' has 0 images in val split.")
        if len(test_files) == 0:
            print(f"Warning: class '{class_name}' has 0 images in test split.")

        copy_files(train_files, TRAIN_DIR / class_name)
        copy_files(val_files, VAL_DIR / class_name)
        copy_files(test_files, TEST_DIR / class_name)

        print(
            f"{class_name}: total={len(image_files)}, "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    save_class_names(class_names, MODELS_DIR / "class_names.json")


def validate_prepared_dataset() -> None:
    validate_directory(DATASET_DIR, "Dataset")
    validate_directory(TRAIN_DIR, "Train")
    validate_directory(VAL_DIR, "Validation")
    validate_directory(TEST_DIR, "Test")

    train_classes = get_class_names(TRAIN_DIR)
    val_classes = get_class_names(VAL_DIR)
    test_classes = get_class_names(TEST_DIR)

    check_same_classes(train_classes, val_classes, test_classes)

    if len(train_classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes in prepared dataset, but found {len(train_classes)}."
        )

    train_counts = count_images_in_split(TRAIN_DIR)
    val_counts = count_images_in_split(VAL_DIR)
    test_counts = count_images_in_split(TEST_DIR)

    print("\nDataset successfully validated.")
    print(f"Dataset root: {DATASET_DIR}")
    print(f"Detected classes: {len(train_classes)}")

    print_split_stats("Train", train_counts)
    print_split_stats("Validation", val_counts)
    print_split_stats("Test", test_counts)


def main() -> None:
    print("Preparing dataset...")
    random.seed(SEED)

    prepare_split()
    validate_prepared_dataset()

    print("\nPreparation completed successfully.")


if __name__ == "__main__":
    main()