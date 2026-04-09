from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "extracted_landmarks"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "splits" / "landmark_split.csv"


def to_repo_relative(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/val/test split manifest from extracted landmark files."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be greater than zero.")
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must add up to 1.0, got {ratio_sum:.6f}.")


def collect_samples(source_dir: Path) -> dict[str, list[Path]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    samples_by_label: dict[str, list[Path]] = {}
    for label_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        samples = sorted(path for path in label_dir.glob("*.npy") if path.is_file())
        if samples:
            samples_by_label[label_dir.name] = samples

    if not samples_by_label:
        raise ValueError(f"No landmark files were found under: {source_dir}")

    return samples_by_label


def compute_split_counts(
    total: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> dict[str, int]:
    counts = {
        "train": max(1, round(total * train_ratio)),
        "val": max(1, round(total * val_ratio)),
        "test": max(1, round(total * test_ratio)),
    }

    while sum(counts.values()) > total:
        split_to_reduce = max(counts, key=lambda split: counts[split])
        if counts[split_to_reduce] == 1:
            break
        counts[split_to_reduce] -= 1

    while sum(counts.values()) < total:
        split_to_increase = min(counts, key=lambda split: counts[split])
        counts[split_to_increase] += 1

    if min(counts.values()) <= 0:
        raise ValueError(f"Invalid split counts for class size {total}: {counts}")

    return counts


def build_manifest_rows(
    samples_by_label: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str | int]]:
    rng = random.Random(seed)
    label_names = sorted(samples_by_label)
    label_map = {label: index for index, label in enumerate(label_names)}
    manifest_rows: list[dict[str, str | int]] = []

    for label in label_names:
        samples = list(samples_by_label[label])
        rng.shuffle(samples)
        counts = compute_split_counts(len(samples), train_ratio, val_ratio, test_ratio)
        train_end = counts["train"]
        val_end = counts["train"] + counts["val"]

        for index, sample_path in enumerate(samples):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"

            manifest_rows.append(
                {
                    "path": sample_path.relative_to(REPO_ROOT).as_posix(),
                    "label": label,
                    "label_id": label_map[label],
                    "split": split,
                }
            )

    return sorted(
        manifest_rows,
        key=lambda row: (str(row["split"]), str(row["label"]), str(row["path"])),
    )


def write_manifest(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "label_id", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]]) -> None:
    split_counts = Counter(str(row["split"]) for row in rows)
    per_label_split_counts = Counter(
        (str(row["label"]), str(row["split"])) for row in rows
    )

    print("Created split manifest with the following counts:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {split_counts[split_name]}")

    print("\nPer-class counts:")
    for label in sorted({str(row["label"]) for row in rows}):
        counts = {
            split_name: per_label_split_counts[(label, split_name)]
            for split_name in ("train", "val", "test")
        }
        print(
            f"  {label}: train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    samples_by_label = collect_samples(args.source_dir)
    rows = build_manifest_rows(
        samples_by_label=samples_by_label,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_manifest(rows, args.output)
    print_summary(rows)
    print(f"\nSaved manifest to: {to_repo_relative(args.output)}")


if __name__ == "__main__":
    main()
