from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import ranking_metrics


def fit_item_popularity(train_csv: str | Path, smoothing: float = 20.0) -> tuple[pd.DataFrame, float, int]:
    """Fit a smoothed click-rate baseline using training interactions only."""
    if smoothing < 0:
        raise ValueError("smoothing cannot be negative")
    aggregates: list[pd.DataFrame] = []
    total_clicks = 0
    total_rows = 0
    for chunk in pd.read_csv(train_csv, usecols=["video_id", "is_click"], chunksize=250_000):
        if chunk["is_click"].isna().any() or not chunk["is_click"].isin([0, 1]).all():
            raise ValueError("Training is_click must be non-null and binary")
        grouped = chunk.groupby("video_id", sort=False)["is_click"].agg(["sum", "count"])
        aggregates.append(grouped)
        total_clicks += int(chunk["is_click"].sum())
        total_rows += len(chunk)
    if total_rows == 0:
        raise ValueError("Training CSV is empty")
    combined = pd.concat(aggregates).groupby(level=0).sum()
    global_ctr = total_clicks / total_rows
    combined["score"] = (combined["sum"] + smoothing * global_ctr) / (combined["count"] + smoothing)
    model = combined.reset_index().rename(columns={"sum": "clicks", "count": "impressions"})
    return model, global_ctr, total_rows


def evaluate_item_popularity(
    validation_csv: str | Path, model: pd.DataFrame, cold_start_score: float
) -> tuple[dict[str, float], int]:
    scores_by_item = model.set_index("video_id")["score"]
    users: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    row_count = 0
    for chunk in pd.read_csv(validation_csv, usecols=["user_id", "video_id", "is_click"], chunksize=250_000):
        users.append(chunk["user_id"].to_numpy())
        labels.append(chunk["is_click"].to_numpy(dtype=np.float32))
        scores.append(chunk["video_id"].map(scores_by_item).fillna(cold_start_score).to_numpy(dtype=np.float64))
        row_count += len(chunk)
    if row_count == 0:
        raise ValueError("Validation CSV is empty")
    metrics = ranking_metrics(np.concatenate(users), np.concatenate(labels), np.concatenate(scores))
    return metrics, row_count


def evaluate_catalog_popularity(
    validation_csv: str | Path,
    model: pd.DataFrame,
    cold_start_score: float,
    item_features_csv: str | Path | None = None,
) -> dict[str, float]:
    """Rank the complete observed catalog and evaluate unique clicked items per user."""
    relevant: dict[int, set[int]] = {}
    catalog = set(model["video_id"].astype(int).tolist())
    if item_features_csv is not None:
        catalog.update(pd.read_csv(item_features_csv, usecols=["video_id"])["video_id"].astype(int).tolist())
    for chunk in pd.read_csv(validation_csv, usecols=["user_id", "video_id", "is_click"], chunksize=250_000):
        catalog.update(chunk["video_id"].astype(int).unique().tolist())
        clicked = chunk[chunk["is_click"] == 1]
        for user, items in clicked.groupby("user_id")["video_id"]:
            relevant.setdefault(int(user), set()).update(items.astype(int).tolist())
    score_map = model.set_index("video_id")["score"]
    catalog_array = np.array(sorted(catalog), dtype=np.int64)
    catalog_scores = pd.Series(catalog_array).map(score_map).fillna(cold_start_score).to_numpy()
    ranked_catalog = catalog_array[np.lexsort((catalog_array, -catalog_scores))]
    top10 = ranked_catalog[:10]
    top50 = set(ranked_catalog[:50].tolist())
    discounts = 1.0 / np.log2(np.arange(len(top10)) + 2.0)
    ndcgs: list[float] = []
    recalls: list[float] = []
    for positives in relevant.values():
        gains = np.fromiter((int(item in positives) for item in top10), dtype=np.float64)
        dcg = float(np.sum(gains * discounts))
        ideal_length = min(len(positives), 10)
        idcg = float(np.sum(1.0 / np.log2(np.arange(ideal_length) + 2.0)))
        ndcgs.append(dcg / idcg)
        recalls.append(len(positives & top50) / len(positives))
    return {
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall@50": float(np.mean(recalls)) if recalls else 0.0,
        "evaluated_users": len(ndcgs),
        "catalog_items": len(catalog_array),
    }


def run_popularity_baseline(
    data_root: str | Path = "data",
    output_dir: str | Path = "artifacts/baselines/item_popularity",
    smoothing: float = 20.0,
    official_ndcg: float | None = None,
    official_recall: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = Path(data_root) / "prepared"
    train_csv = prepared / "train.csv"
    validation_csv = prepared / "validation.csv"
    if not train_csv.is_file() or not validation_csv.is_file():
        raise FileNotFoundError("Prepared train/validation files are missing; run prepare-kuairand first")
    model, global_ctr, train_rows = fit_item_popularity(train_csv, smoothing)
    logged_metrics, validation_rows = evaluate_item_popularity(validation_csv, model, global_ctr)
    item_features = prepared / "video_features_basic_pure.csv"
    catalog_metrics = evaluate_catalog_popularity(
        validation_csv,
        model,
        global_ctr,
        item_features if item_features.is_file() else None,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.csv"
    temporary = model_path.with_suffix(".csv.tmp")
    model.to_csv(temporary, index=False)
    os.replace(temporary, model_path)
    result: dict[str, Any] = {
        "baseline": "smoothed_item_popularity",
        "label": "is_click",
        "selection_metrics": ["ndcg@10", "recall@50"],
        "smoothing": smoothing,
        "global_train_ctr": global_ctr,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "validation_metrics": logged_metrics,
        "evaluation_protocol": "logged validation impressions (challenge contract pending organizer evaluator)",
        "full_catalog_validation_metrics": catalog_metrics,
        "model_path": str(model_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if official_ndcg is not None and official_recall is not None:
        deltas = {
            "ndcg@10": logged_metrics["ndcg@10"] - official_ndcg,
            "recall@50": logged_metrics["recall@50"] - official_recall,
        }
        result["official_reference"] = {"ndcg@10": official_ndcg, "recall@50": official_recall}
        result["deltas"] = deltas
        result["mean_absolute_improvement"] = (deltas["ndcg@10"] + deltas["recall@50"]) / 2.0
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
