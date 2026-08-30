from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .features import TEMPORAL_COLUMNS, add_temporal_features, merge_side_features


class InteractionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, users: np.ndarray, items: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.users = np.asarray(users)
        self.items = np.asarray(items)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class PairwiseInteractionDataset(Dataset):
    """Deterministic clicked/non-clicked pairs drawn only from each user's logged exposures."""

    def __init__(self, base: InteractionDataset, negatives_per_positive: int, seed: int):
        if base.labels.ndim != 1:
            raise ValueError("PairwiseInteractionDataset requires one-dimensional click labels")
        rng = np.random.default_rng(seed)
        positive_indices: list[int] = []
        negative_indices: list[int] = []
        order = np.argsort(base.users, kind="stable")
        sorted_users = base.users[order]
        boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
        labels = base.labels.numpy()
        for group_indices in np.split(order, boundaries):
            positives = group_indices[labels[group_indices] == 1]
            negatives = group_indices[labels[group_indices] == 0]
            if len(positives) == 0 or len(negatives) == 0:
                continue
            sampled = rng.choice(negatives, size=(len(positives), negatives_per_positive), replace=True)
            positive_indices.extend(np.repeat(positives, negatives_per_positive).tolist())
            negative_indices.extend(sampled.reshape(-1).tolist())
        if not positive_indices:
            raise ValueError("No users contain both positive and negative logged impressions")
        self.features = base.features
        self.positive_indices = torch.as_tensor(positive_indices, dtype=torch.long)
        self.negative_indices = torch.as_tensor(negative_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.positive_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[self.positive_indices[index]], self.features[self.negative_indices[index]]


def _read_optional(path: str | None, key: str, requested: set[str]) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_csv(path, usecols=lambda column: column == key or column in requested)


def load_interactions(config: DataConfig, split: str) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("Only train and validation data may be loaded during development")
    path = config.train_csv if split == "train" else config.validation_csv
    limit = config.max_train_rows if split == "train" else config.max_validation_rows
    if not Path(path).is_file():
        raise FileNotFoundError(f"{split} CSV not found: {path}")
    requested_features = set(config.categorical_features)
    interaction_columns = requested_features | {
        config.user_column,
        config.item_column,
        config.label_column,
    } | set(config.auxiliary_label_columns)
    if config.temporal_features:
        interaction_columns |= {"time_ms", "date"}
    frame = pd.read_csv(path, nrows=limit, usecols=lambda column: column in interaction_columns)
    required = {config.user_column, config.item_column, config.label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = merge_side_features(
        frame,
        _read_optional(config.user_features_csv, config.user_column, requested_features),
        _read_optional(config.item_features_csv, config.item_column, requested_features),
        config.user_column,
        config.item_column,
    )
    if config.temporal_features:
        frame = add_temporal_features(frame)
    for label_column in [config.label_column, *config.auxiliary_label_columns]:
        labels = frame[label_column]
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{label_column} must contain non-null binary labels")
    if frame.empty:
        raise ValueError(f"{split} data is empty")
    return frame


def load_scoring_interactions(config: DataConfig, path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Load features for inference without reading or requiring outcome labels."""
    scoring_path = Path(path)
    if not scoring_path.is_file():
        raise FileNotFoundError(f"scoring CSV not found: {scoring_path}")
    requested_features = set(config.categorical_features)
    interaction_columns = requested_features | {config.user_column, config.item_column}
    if config.temporal_features:
        interaction_columns |= {"time_ms", "date"}
    frame = pd.read_csv(scoring_path, nrows=limit, usecols=lambda column: column in interaction_columns)
    required = {config.user_column, config.item_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{scoring_path} is missing required columns: {sorted(missing)}")
    frame = merge_side_features(
        frame,
        _read_optional(config.user_features_csv, config.user_column, requested_features),
        _read_optional(config.item_features_csv, config.item_column, requested_features),
        config.user_column,
        config.item_column,
    )
    if config.temporal_features:
        frame = add_temporal_features(frame)
    if frame.empty:
        raise ValueError("scoring data is empty")
    return frame


def configured_feature_columns(config: DataConfig) -> list[str]:
    columns = list(config.categorical_features)
    if config.temporal_features:
        columns.extend(column for column in TEMPORAL_COLUMNS if column not in columns)
    return columns


def sample_logged_negatives(
    frame: pd.DataFrame, label_column: str, negative_ratio: float | None, seed: int
) -> pd.DataFrame:
    """Keep all positives and sample only from observed non-clicked impressions."""
    if negative_ratio is None:
        return frame
    positives = frame[frame[label_column] == 1]
    negatives = frame[frame[label_column] == 0]
    desired = min(len(negatives), int(round(len(positives) * negative_ratio)))
    if desired == 0:
        return positives.reset_index(drop=True)
    sampled = negatives.sample(n=desired, random_state=seed, replace=False)
    return pd.concat([positives, sampled], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
