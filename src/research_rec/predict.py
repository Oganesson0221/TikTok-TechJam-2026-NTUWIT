from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import config_from_dict
from .data import load_scoring_interactions
from .features import CategoricalFeatureEncoder
from .models import build_model
from .training import resolve_device


def _encoder_from_state(state: dict[str, Any]) -> CategoricalFeatureEncoder:
    encoder = CategoricalFeatureEncoder(list(state["columns"]))
    encoder.vocabularies = state["vocabularies"]
    return encoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def export_predictions(
    checkpoint_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    top_k: int | None = None,
    batch_size: int = 4096,
    device_name: str = "auto",
    limit: int | None = None,
    column_mapping: dict[str, str] | None = None,
    output_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Export a schema-neutral ranked CSV without inspecting outcome labels."""
    checkpoint_file = Path(checkpoint_path)
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    encoder = _encoder_from_state(checkpoint["feature_encoder"])
    frame = load_scoring_interactions(config.data, input_csv, limit=limit)
    encoded = torch.as_tensor(encoder.transform(frame), dtype=torch.long)
    task_count = 1 + len(config.data.auxiliary_label_columns)
    model = build_model(config.model, checkpoint["field_dims"], task_count=task_count).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    requested_batch_size = batch_size
    effective_batch_size = batch_size
    recoveries: list[str] = []
    while True:
        predictions: list[np.ndarray] = []
        loader = DataLoader(TensorDataset(encoded), batch_size=effective_batch_size, shuffle=False)
        try:
            for (features,) in loader:
                logits = model(features.to(device))
                primary_logits = logits[:, 0] if logits.ndim == 2 else logits
                predictions.append(primary_logits.sigmoid().cpu().numpy())
            break
        except RuntimeError as error:
            is_cuda_blas_error = "CUBLAS_STATUS_NOT_INITIALIZED" in str(error)
            if device.type != "cuda" or not is_cuda_blas_error or effective_batch_size <= 1024:
                raise
            previous_batch_size = effective_batch_size
            effective_batch_size = max(1024, effective_batch_size // 2)
            recoveries.append(
                f"Reduced inference batch_size from {previous_batch_size} to "
                f"{effective_batch_size} after CUDA BLAS initialization failure"
            )
            torch.cuda.empty_cache()

    user_column = config.data.user_column
    item_column = config.data.item_column
    output = frame[[user_column, item_column]].copy()
    output["score"] = np.concatenate(predictions)
    output["rank"] = output.groupby(user_column)["score"].rank(method="first", ascending=False).astype("int64")
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        output = output[output["rank"] <= top_k]
    output = output.sort_values([user_column, "rank"], kind="stable")
    available_columns = [user_column, item_column, "score", "rank"]
    selected_columns = output_columns or available_columns
    unknown_columns = set(selected_columns) - set(available_columns)
    if unknown_columns:
        raise ValueError(f"Unknown output columns: {sorted(unknown_columns)}")
    mapping = column_mapping or {}
    unknown_mapping = set(mapping) - set(available_columns)
    if unknown_mapping:
        raise ValueError(f"Unknown column mapping keys: {sorted(unknown_mapping)}")
    output = output[selected_columns].rename(columns=mapping)

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(destination)
    summary = {
        "checkpoint": str(checkpoint_file.resolve()),
        "input": str(Path(input_csv).resolve()),
        "output": str(destination.resolve()),
        "rows": len(output),
        "users": int(frame[user_column].nunique()),
        "top_k": top_k,
        "columns": list(output.columns),
        "column_mapping": mapping,
        "sha256": _sha256(destination),
        "label_columns_read": [],
        "schema_status": "generic; rename/map columns when the organizer schema is published",
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": effective_batch_size,
        "recoveries": recoveries,
    }
    destination.with_suffix(destination.suffix + ".meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
