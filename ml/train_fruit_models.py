"""Training and prediction utilities for fruit maturity and quality models.

Dataset layout expected by this script:

<root>
├── Maturity/
│   ├── Immature/
│   └── Mature/
└── Quality/
    ├── Fresh/
    └── Rotten/

Use torchvision transforms so variable-size images are supported.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


@dataclass
class DataConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_split: float = 0.8
    num_workers: int = 4
    seed: int = 42
    use_pretrained: bool = True


TASK_INFO = {
    "maturity": {
        "subdir": "Maturity",
        "classes": ["Immature", "Mature"],
        "checkpoint": "maturity_best-3-1.pth",
    },
    "quality": {
        "subdir": "Quality",
        "classes": ["Fresh", "Rotten"],
        "checkpoint": "quality_best-3-1.pth",
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    predict_transform = eval_transform
    return train_transform, eval_transform, predict_transform


def split_dataset(dataset: datasets.ImageFolder, train_split: float, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloaders(cfg: DataConfig, task: str) -> Tuple[DataLoader, DataLoader, Dict[int, int]]:
    task_cfg = TASK_INFO[task]
    train_transform, eval_transform, _ = build_transforms()

    base_dataset = datasets.ImageFolder(cfg.data_root / task_cfg["subdir"], transform=train_transform)
    train_set, val_set = split_dataset(base_dataset, cfg.train_split, cfg.seed)

    val_set.dataset = datasets.ImageFolder(cfg.data_root / task_cfg["subdir"], transform=eval_transform)

    class_counts = Counter([label for _, label in base_dataset])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, dict(class_counts)


def build_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def class_weight_tensor(class_counts: Dict[int, int], device: torch.device) -> torch.Tensor:
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(class_counts))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_acc += compute_accuracy(outputs, labels) * images.size(0)

    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)


def train_one_task(cfg: DataConfig, task: str, device: torch.device) -> Path:
    seed_everything(cfg.seed)
    task_cfg = TASK_INFO[task]
    train_loader, val_loader, counts = build_dataloaders(cfg, task)

    model = build_model(num_classes=len(task_cfg["classes"]), use_pretrained=cfg.use_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(counts, device))
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_acc = 0.0
    best_path = cfg.output_dir / task_cfg["checkpoint"]
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_acc += compute_accuracy(outputs.detach(), labels) * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"[{task}] Epoch {epoch}/{cfg.epochs} "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": task_cfg["classes"],
                    "task": task,
                },
                best_path,
            )
            print(f"[{task}] New best model saved to {best_path} (val acc={val_acc:.4f})")

    return best_path


def predict(image_path: Path, checkpoint: Path, device: torch.device) -> Tuple[str, Dict[str, float]]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location=device)
    classes: Iterable[str] = payload["classes"]
    task: str = payload["task"]
    _, _, predict_transform = build_transforms()

    model = build_model(num_classes=len(classes), use_pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = predict_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    confidence = {cls: probs[idx].item() for idx, cls in enumerate(classes)}
    predicted_class = classes[int(probs.argmax().item())]
    return predicted_class, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fruit maturity and quality classifiers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing Maturity/ and Quality/ folders")
    train_parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to store checkpoints")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--train-split", type=float, default=0.8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining")
    train_parser.add_argument("--task", choices=["maturity", "quality", "both"], default="both")

    predict_parser = subparsers.add_parser("predict", help="Run inference on a single image")
    predict_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved .pth checkpoint")
    predict_parser.add_argument("--image", type=Path, required=True, help="Path to an RGB image")
    predict_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def handle_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed,
        use_pretrained=not args.no_pretrained,
    )

    tasks = [args.task] if args.task != "both" else ["maturity", "quality"]
    for task in tasks:
        print(f"Training task: {task} on device {device}")
        best_checkpoint = train_one_task(cfg, task, device)
        print(f"Finished training {task}. Best checkpoint: {best_checkpoint}")


def handle_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    label, conf = predict(args.image, args.checkpoint, device)
    formatted = ", ".join([f"{k}: {v:.3f}" for k, v in conf.items()])
    print(f"Prediction: {label} ({formatted})")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        handle_train(args)
    elif args.command == "predict":
        handle_predict(args)


if __name__ == "__main__":
    main()
