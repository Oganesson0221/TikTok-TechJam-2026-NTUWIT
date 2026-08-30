from pathlib import Path

import pandas as pd

from research_rec.baseline import evaluate_catalog_popularity, evaluate_item_popularity, fit_item_popularity


def test_popularity_baseline_fits_training_only_and_handles_cold_items(tmp_path: Path):
    train = pd.DataFrame(
        {
            "video_id": [1, 1, 2, 2],
            "is_click": [1, 1, 0, 0],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": [10, 10, 20, 20],
            "video_id": [1, 2, 3, 2],
            "is_click": [1, 0, 1, 0],
        }
    )
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    model, global_ctr, rows = fit_item_popularity(train_path, smoothing=0)
    metrics, validation_rows = evaluate_item_popularity(validation_path, model, global_ctr)
    assert rows == 4
    assert validation_rows == 4
    assert global_ctr == 0.5
    assert metrics["evaluated_users"] == 2
    assert metrics["ndcg@10"] == 1.0
    catalog_metrics = evaluate_catalog_popularity(validation_path, model, global_ctr)
    assert catalog_metrics["catalog_items"] == 3
    assert catalog_metrics["evaluated_users"] == 2
