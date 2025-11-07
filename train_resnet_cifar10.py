"""Train and validate a ResNet model on the CIFAR-10 dataset.

This script provides a reusable training pipeline that downloads CIFAR-10,
performs basic data augmentation, trains a ResNet model, and reports
validation metrics at the end of every epoch.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


@dataclass
class TrainConfig:
    """Configuration options for CIFAR-10 training."""

    data_dir: Path
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 20
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    step_size: int = 60
    gamma: float = 0.2
    resume: Path | None = None
    output_dir: Path = Path("runs")


def get_data_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for CIFAR-10."""

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=str(cfg.data_dir), train=True, download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=str(cfg.data_dir), train=False, download=True, transform=val_transform
    )

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
    return train_loader, val_loader


def build_model() -> nn.Module:
    """Return a ResNet model adapted for CIFAR-10."""

    model = torchvision.models.resnet18(weights=None, num_classes=10)
    # Adjust the first convolution to better suit 32x32 inputs.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, labels).item() * images.size(0)

    num_samples = len(dataloader.dataset)
    return {
        "loss": running_loss / num_samples,
        "acc": running_acc / num_samples,
    }


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels).item() * images.size(0)

    num_samples = len(dataloader.dataset)
    return {
        "loss": running_loss / num_samples,
        "acc": running_acc / num_samples,
    }


def save_checkpoint(state: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0)
    return start_epoch


def run_training(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(cfg)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.gamma
    )

    start_epoch = 0
    if cfg.resume is not None and cfg.resume.exists():
        print(f"Resuming training from checkpoint: {cfg.resume}")
        start_epoch = load_checkpoint(cfg.resume, model, optimizer)
        scheduler.last_epoch = start_epoch - 1

    best_acc = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_metrics = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
        )

        is_best = val_metrics["acc"] > best_acc
        if is_best:
            best_acc = val_metrics["acc"]

        checkpoint_path = cfg.output_dir / "checkpoint.pth"
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
            },
            checkpoint_path,
        )
        if is_best:
            best_path = cfg.output_dir / "best.pth"
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                },
                best_path,
            )

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory to store CIFAR-10 data",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--step-size", type=int, default=60, help="LR scheduler step size"
    )
    parser.add_argument("--gamma", type=float, default=0.2, help="LR scheduler gamma")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory to store checkpoints",
    )

    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        resume=args.resume,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
