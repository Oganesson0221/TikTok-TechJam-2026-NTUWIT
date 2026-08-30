from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import (
    InteractionDataset,
    PairwiseInteractionDataset,
    configured_feature_columns,
    load_interactions,
    sample_logged_negatives,
)
from .features import CategoricalFeatureEncoder
from .metrics import ranking_metrics
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def _atomic_checkpoint(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _make_dataset(frame: Any, encoder: CategoricalFeatureEncoder, config: ExperimentConfig) -> InteractionDataset:
    data = config.data
    label_columns = [data.label_column, *data.auxiliary_label_columns]
    labels = frame[label_columns].to_numpy(dtype=np.float32)
    if len(label_columns) == 1:
        labels = labels[:, 0]
    return InteractionDataset(
        encoder.transform(frame),
        labels,
        frame[data.user_column].to_numpy(),
        frame[data.item_column].to_numpy(),
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, dataset: InteractionDataset, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    predictions: list[np.ndarray] = []
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        primary_logits = logits[:, 0] if logits.ndim == 2 else logits
        primary_labels = labels[:, 0] if labels.ndim == 2 else labels
        total_loss += float(criterion(primary_logits, primary_labels).item())
        predictions.append(primary_logits.sigmoid().cpu().numpy())
    scores = np.concatenate(predictions) if predictions else np.empty(0)
    raw_labels = dataset.labels.numpy()
    primary_labels = raw_labels[:, 0] if raw_labels.ndim == 2 else raw_labels
    metrics = ranking_metrics(dataset.users, primary_labels, scores)
    metrics["loss"] = total_loss / max(len(dataset), 1)
    return metrics


def train_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    started = time.perf_counter()

    full_train = load_interactions(config.data, "train")
    validation = load_interactions(config.data, "validation")
    columns = configured_feature_columns(config.data)
    encoder = CategoricalFeatureEncoder(columns).fit(full_train)
    sampled_train = sample_logged_negatives(
        full_train,
        config.data.label_column,
        config.data.negative_ratio,
        config.training.seed,
    )
    if sampled_train.empty:
        raise ValueError("Training data is empty after negative sampling")
    train_data = _make_dataset(sampled_train, encoder, config)
    validation_data = _make_dataset(validation, encoder, config)
    generator = torch.Generator().manual_seed(config.training.seed)
    optimization_data = (
        PairwiseInteractionDataset(train_data, config.training.pairwise_negatives, config.training.seed)
        if config.training.loss_name in {"bpr", "hybrid"}
        else train_data
    )
    train_loader = DataLoader(
        optimization_data,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    task_count = 1 + len(config.data.auxiliary_label_columns)
    model = build_model(config.model, encoder.field_dims, task_count=task_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = None
    if config.training.lr_scheduler_factor is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.training.lr_scheduler_factor,
            patience=config.training.lr_scheduler_patience,
        )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    if task_count > 1:
        auxiliary_weight = config.model.auxiliary_loss_weight / (task_count - 1)
        task_weights = torch.tensor(
            [1.0 - config.model.auxiliary_loss_weight, *([auxiliary_weight] * (task_count - 1))],
            device=device,
        )
    else:
        task_weights = None
    stopper = EarlyStopping(config.training.patience, config.training.min_delta)
    checkpoint = Path(config.training.checkpoint_dir) / config.training.experiment_name / "best.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for first, second in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if config.training.loss_name == "bce":
                features, labels = first.to(device), second.to(device)
                logits = model(features)
                element_losses = criterion(logits, labels)
                if task_weights is None:
                    loss = element_losses.mean()
                else:
                    loss = (element_losses * task_weights).sum(dim=1).mean()
                example_count = len(labels)
            else:
                positive_features, negative_features = first.to(device), second.to(device)
                positive_logits = model(positive_features)
                negative_logits = model(negative_features)
                pairwise_loss = torch.nn.functional.softplus(-(positive_logits - negative_logits)).mean()
                if config.training.loss_name == "hybrid":
                    positive_bce = criterion(positive_logits, torch.ones_like(positive_logits)).mean()
                    negative_bce = criterion(negative_logits, torch.zeros_like(negative_logits)).mean()
                    pointwise_loss = 0.5 * (positive_bce + negative_bce)
                    weight = config.training.hybrid_bce_weight
                    loss = (1.0 - weight) * pairwise_loss + weight * pointwise_loss
                else:
                    loss = pairwise_loss
                example_count = len(positive_features)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * example_count

        metrics = evaluate(model, validation_loader, validation_data, device)
        metrics["train_loss"] = running_loss / max(len(optimization_data), 1)
        metrics["epoch"] = epoch
        selection_score = (float(metrics["ndcg@10"]) + float(metrics["recall@50"])) / 2.0
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(metrics)
        if stopper.update(selection_score):
            _atomic_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "feature_encoder": encoder.state_dict(),
                    "field_dims": encoder.field_dims,
                    "epoch": epoch,
                    "metrics": metrics,
                    "selection_score": selection_score,
                },
                checkpoint,
            )
        if scheduler is not None:
            scheduler.step(selection_score)
        if stopper.should_stop:
            break

    best = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    best_metrics = evaluate(model, validation_loader, validation_data, device)
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "experiment_name": config.training.experiment_name,
        "model": config.model.name,
        "device": str(device),
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "epochs_completed": len(history),
        "best_epoch": best["epoch"],
        "early_stopped": len(history) < config.training.epochs,
        "best_metrics": best_metrics,
        "checkpoint": str(checkpoint.resolve()),
        "elapsed_seconds": elapsed,
        "history": history,
    }
    summary_path = checkpoint.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
