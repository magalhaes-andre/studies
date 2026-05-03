from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .annotations import load_annotations
from .dataset import LungTumorCropDataset
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    probabilities: list[float] = []
    labels: list[int] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)
        probs = torch.sigmoid(logits)

        losses.append(loss.item())
        probabilities.extend(probs.squeeze(1).cpu().tolist())
        labels.extend(targets.squeeze(1).int().cpu().tolist())

    predictions = [1 if p >= 0.5 else 0 for p in probabilities]
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(labels, predictions)) if labels else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)) if labels else 0.0,
        "recall": float(recall_score(labels, predictions, zero_division=0)) if labels else 0.0,
    }

    if len(set(labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module, optimizer: Adam) -> float:
    model.train()
    losses: list[float] = []

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def split_by_patient(annotations: pd.DataFrame, seed: int, val_size: float, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = annotations["patient_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(annotations, groups=groups))

    train_val = annotations.iloc[train_val_idx].reset_index(drop=True)
    test = annotations.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / max(1e-8, 1.0 - test_size)
    splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed + 1)
    train_idx, val_idx = next(splitter.split(train_val, groups=train_val["patient_id"]))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal CT tumor-vs-background classifier.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=350.0)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(args.annotations_dir, args.image_root)
    annotations.to_csv(output_dir / "annotations_resolved.csv", index=False)

    train_df, val_df, test_df = split_by_patient(annotations, args.seed, args.val_size, args.test_size)
    train_df.assign(split="train").to_csv(output_dir / "train_annotations.csv", index=False)
    val_df.assign(split="val").to_csv(output_dir / "val_annotations.csv", index=False)
    test_df.assign(split="test").to_csv(output_dir / "test_annotations.csv", index=False)

    train_dataset = LungTumorCropDataset(
        train_df,
        image_size=args.img_size,
        negatives_per_positive=args.negatives_per_positive,
        window_center=args.window_center,
        window_width=args.window_width,
        padding=args.padding,
        augment=True,
        seed=args.seed,
    )
    val_dataset = LungTumorCropDataset(
        val_df,
        image_size=args.img_size,
        negatives_per_positive=args.negatives_per_positive,
        window_center=args.window_center,
        window_width=args.window_width,
        padding=args.padding,
        augment=False,
        seed=args.seed + 1,
    )
    test_dataset = LungTumorCropDataset(
        test_df,
        image_size=args.img_size,
        negatives_per_positive=args.negatives_per_positive,
        window_center=args.window_center,
        window_width=args.window_width,
        padding=args.padding,
        augment=False,
        seed=args.seed + 2,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    best_auc = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        candidate_auc = val_metrics["roc_auc"]
        if np.isnan(candidate_auc):
            candidate_auc = val_metrics["accuracy"]

        if candidate_auc > best_auc:
            best_auc = float(candidate_auc)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, loss_fn)

    summary = {
        "train_annotations": int(len(train_df)),
        "val_annotations": int(len(val_df)),
        "test_annotations": int(len(test_df)),
        "history": history,
        "test_metrics": test_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Saved checkpoint to", output_dir / "best.pt")
    print(json.dumps({"test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
