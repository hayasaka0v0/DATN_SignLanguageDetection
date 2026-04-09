from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "splits" / "landmark_split.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "baseline_mlp"
EXPECTED_SAMPLE_SHAPE = (30, 258)


def to_repo_relative(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple MLP baseline on flattened landmark arrays."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest file is empty: {manifest_path}")
    return rows


def load_splits(
    rows: list[dict[str, str]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[int, str]]:
    features_by_split: dict[str, list[np.ndarray]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    labels_by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    id_to_label: dict[int, str] = {}

    for row in rows:
        split = row["split"]
        if split not in features_by_split:
            raise ValueError(f"Unexpected split value in manifest: {split}")

        sample_path = REPO_ROOT / row["path"]
        sample = np.load(sample_path)
        if sample.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"Expected sample shape {EXPECTED_SAMPLE_SHAPE}, got {sample.shape} for {sample_path}"
            )

        label_id = int(row["label_id"])
        id_to_label[label_id] = row["label"]
        features_by_split[split].append(sample.astype(np.float32).reshape(-1))
        labels_by_split[split].append(label_id)

    arrays_by_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_dim = EXPECTED_SAMPLE_SHAPE[0] * EXPECTED_SAMPLE_SHAPE[1]
    for split in ("train", "val", "test"):
        if not features_by_split[split]:
            raise ValueError(f"Manifest contains no rows for split: {split}")
        features = np.stack(features_by_split[split], axis=0)
        labels = np.asarray(labels_by_split[split], dtype=np.int64)
        if features.shape[1] != feature_dim:
            raise ValueError(
                f"Unexpected feature dimension in split {split}: {features.shape[1]}"
            )
        arrays_by_split[split] = (features, labels)

    return arrays_by_split, dict(sorted(id_to_label.items()))


def normalize_splits(
    arrays_by_split: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_features, _ = arrays_by_split["train"]
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, (features, labels) in arrays_by_split.items():
        normalized[split] = ((features - feature_mean) / feature_std, labels)
    return normalized


def build_dataloaders(
    arrays_by_split: dict[str, tuple[np.ndarray, np.ndarray]], batch_size: int
) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = {}
    for split, (features, labels) in arrays_by_split.items():
        dataset = TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).long(),
        )
        dataloaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train")
        )
    return dataloaders


class LandmarkMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = loss_fn(logits, targets)

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_value, predicted_value in zip(targets, predictions, strict=False):
        confusion[int(true_value), int(predicted_value)] += 1

    accuracy = float((predictions == targets).mean())
    average_loss = total_loss / max(total_samples, 1)
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    learning_rate: float,
    epochs: int,
    device: torch.device,
    num_classes: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for features, targets in dataloaders["train"]:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

        train_loss = total_loss / max(total_samples, 1)
        val_metrics = evaluate_model(
            model, dataloaders["val"], loss_fn, device, num_classes
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        if float(val_metrics["accuracy"]) >= best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={float(val_metrics['loss']):.4f} | "
            f"val_accuracy={float(val_metrics['accuracy']):.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    return model, history


def save_outputs(
    output_dir: Path, model: nn.Module, metrics: dict[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    rows = read_manifest(args.manifest)
    arrays_by_split, label_map = load_splits(rows)
    arrays_by_split = normalize_splits(arrays_by_split)
    dataloaders = build_dataloaders(arrays_by_split, batch_size=args.batch_size)

    input_dim = EXPECTED_SAMPLE_SHAPE[0] * EXPECTED_SAMPLE_SHAPE[1]
    model = LandmarkMLP(input_dim=input_dim, num_classes=len(label_map)).to(device)
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=device,
        num_classes=len(label_map),
    )

    loss_fn = nn.CrossEntropyLoss()
    val_metrics = evaluate_model(
        model, dataloaders["val"], loss_fn, device, len(label_map)
    )
    test_metrics = evaluate_model(
        model, dataloaders["test"], loss_fn, device, len(label_map)
    )
    metrics = {
        "manifest": to_repo_relative(args.manifest),
        "device": str(device),
        "input_shape": list(EXPECTED_SAMPLE_SHAPE),
        "flattened_input_dim": input_dim,
        "label_map": {
            str(label_id): label_name for label_id, label_name in label_map.items()
        },
        "history": history,
        "best_validation_accuracy": max(epoch["val_accuracy"] for epoch in history),
        "validation": {
            "loss": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "confusion_matrix": val_metrics["confusion_matrix"],
        },
        "test": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }
    save_outputs(args.output_dir, model, metrics)

    print("\nTraining complete.")
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Saved outputs to: {to_repo_relative(args.output_dir)}")


if __name__ == "__main__":
    main()
