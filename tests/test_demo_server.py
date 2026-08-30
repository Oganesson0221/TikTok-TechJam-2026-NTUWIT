import csv
from pathlib import Path

from research_rec.demo_server import DemoDataStore


def test_demo_data_store_returns_ranked_rows_and_sample_users(tmp_path: Path):
    prediction_path = tmp_path / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "video_id", "score", "rank"])
        writer.writeheader()
        writer.writerows(
            [
                {"user_id": 7, "video_id": 12, "score": 0.3, "rank": 2},
                {"user_id": 7, "video_id": 11, "score": 0.8, "rank": 1},
                {"user_id": 8, "video_id": 20, "score": 0.4, "rank": 1},
            ]
        )
    store = DemoDataStore(prediction_path)
    assert [row["video_id"] for row in store.ranking("7")] == [11, 12]
    assert store.sample_users(1) == [{"user_id": "7", "candidates": 2}]
    assert store.ranking("missing") == []
