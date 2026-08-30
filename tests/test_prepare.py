import csv
from pathlib import Path

import pandas as pd

from research_rec.prepare import split_late_interactions, summarize_split


def test_fixed_row_half_split_and_summary(tmp_path: Path):
    source = tmp_path / "late.csv"
    rows = [
        [1, 10, 100, 1],
        [1, 11, 101, 0],
        [2, 12, 102, 1],
        [2, 13, 103, 0],
        [3, 14, 104, 1],
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "video_id", "time_ms", "is_click"])
        writer.writerows(rows)
    validation = tmp_path / "validation.csv"
    test = tmp_path / "test.csv"
    counts = split_late_interactions(source, validation, test)
    assert counts == (2, 3)
    assert pd.read_csv(validation)["video_id"].tolist() == [10, 11]
    assert pd.read_csv(test)["video_id"].tolist() == [12, 13, 14]
    summary = summarize_split(validation)
    assert summary.rows == 2
    assert summary.positives == 1
    assert summary.min_time_ms == 100

