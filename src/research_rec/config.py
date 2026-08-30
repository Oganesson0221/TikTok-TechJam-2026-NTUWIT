from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json

try:
    import yaml
except ModuleNotFoundError:  # JSON configs keep cluster execution dependency-free.
    yaml = None


@dataclass
class DataConfig:
    train_csv: str = "data/train.csv"
    validation_csv: str = "data/validation.csv"
    user_features_csv: str | None = None
    item_features_csv: str | None = None
    label_column: str = "is_click"
    auxiliary_label_columns: list[str] = field(default_factory=list)
    user_column: str = "user_id"
    item_column: str = "video_id"
    categorical_features: list[str] = field(default_factory=lambda: ["user_id", "video_id"])
    temporal_features: bool = False
    negative_ratio: float | None = None
    max_train_rows: int | None = None
    max_validation_rows: int | None = None


@dataclass
class ModelConfig:
    name: str = "deepfm"
    embedding_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    cross_layers: int = 3
    auxiliary_loss_weight: float = 0.25


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 4
    min_delta: float = 1e-5
    lr_scheduler_factor: float | None = None
    lr_scheduler_patience: int = 2
    loss_name: str = "bce"
    pairwise_negatives: int = 2
    hybrid_bce_weight: float = 0.25
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "artifacts/checkpoints"
    experiment_name: str = "deepfm_default"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.model.name.lower() not in {"mf", "deepfm", "dcn", "multitask_dcn"}:
            raise ValueError("model.name must be one of: mf, deepfm, dcn, multitask_dcn")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.training.patience < 1:
            raise ValueError("patience must be positive")
        if self.training.lr_scheduler_factor is not None and not 0 < self.training.lr_scheduler_factor < 1:
            raise ValueError("lr_scheduler_factor must be between 0 and 1")
        if self.training.lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience cannot be negative")
        if self.training.loss_name not in {"bce", "bpr", "hybrid"}:
            raise ValueError("loss_name must be bce, bpr, or hybrid")
        if self.training.pairwise_negatives < 1:
            raise ValueError("pairwise_negatives must be positive")
        if not 0 <= self.training.hybrid_bce_weight <= 1:
            raise ValueError("hybrid_bce_weight must be between 0 and 1")
        if self.data.negative_ratio is not None and self.data.negative_ratio < 0:
            raise ValueError("negative_ratio cannot be negative")
        if not 0 <= self.model.auxiliary_loss_weight <= 1:
            raise ValueError("auxiliary_loss_weight must be between 0 and 1")
        if self.model.name.lower() == "multitask_dcn" and not self.data.auxiliary_label_columns:
            raise ValueError("multitask_dcn requires data.auxiliary_label_columns")
        if self.model.name.lower() != "multitask_dcn" and self.data.auxiliary_label_columns:
            raise ValueError("auxiliary labels are supported only by multitask_dcn")
        if self.model.name.lower() == "multitask_dcn" and self.training.loss_name != "bce":
            raise ValueError("multitask_dcn currently requires loss_name=bce")
        required = {self.data.user_column, self.data.item_column}
        if not required.issubset(self.data.categorical_features):
            raise ValueError("categorical_features must contain the user and item columns")
        if self.model.name.lower() == "mf" and self.data.categorical_features[:2] != [
            self.data.user_column,
            self.data.item_column,
        ]:
            raise ValueError("mf requires user and item to be the first two categorical_features")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    unknown = set(values) - set(instance.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    unknown = set(raw) - {"data", "model", "training"}
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    config = ExperimentConfig(
        data=_merge_dataclass(DataConfig(), raw.get("data", {})),
        model=_merge_dataclass(ModelConfig(), raw.get("model", {})),
        training=_merge_dataclass(TrainingConfig(), raw.get("training", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(load_structured_file(path))


def apply_override(config: ExperimentConfig, expression: str) -> None:
    """Apply a validated YAML scalar/list override in section.key=value form."""
    if "=" not in expression:
        raise ValueError(f"Override must use section.key=value syntax: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    parts = dotted_key.split(".")
    if len(parts) != 2 or parts[0] not in {"data", "model", "training"}:
        raise ValueError(f"Invalid override path: {dotted_key}")
    section: Any = getattr(config, parts[0])
    if not hasattr(section, parts[1]):
        raise ValueError(f"Unknown override: {dotted_key}")
    if yaml is not None:
        value = yaml.safe_load(raw_value)
    else:
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
    setattr(section, parts[1], value)


def load_structured_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            if yaml is None:
                raise RuntimeError(f"PyYAML is required to read {config_path}; use an equivalent JSON config")
            raw = yaml.safe_load(handle)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return raw
