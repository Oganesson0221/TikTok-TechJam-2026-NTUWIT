import numpy as np
import pandas as pd
import torch

from research_rec.data import InteractionDataset, PairwiseInteractionDataset, sample_logged_negatives
from research_rec.features import CategoricalFeatureEncoder, add_temporal_features


def test_encoder_reserves_zero_for_unseen_values():
    train = pd.DataFrame({"user_id": [10, 20], "video_id": [1, 2]})
    validation = pd.DataFrame({"user_id": [10, 999], "video_id": [3, 2]})
    encoder = CategoricalFeatureEncoder(["user_id", "video_id"]).fit(train)
    encoded = encoder.transform(validation)
    assert encoded.tolist() == [[1, 0], [0, 2]]
    assert encoder.field_dims == [3, 3]


def test_temporal_features_from_kuairand_timestamp():
    timestamp = pd.Timestamp("2022-04-09 15:30:00").timestamp() * 1000
    result = add_temporal_features(pd.DataFrame({"time_ms": [timestamp]}))
    assert result.loc[0, "hour"] == 15
    assert result.loc[0, "day_of_week"] == 5
    assert result.loc[0, "is_weekend"] == 1


def test_negative_sampling_uses_logged_rows_and_is_reproducible():
    frame = pd.DataFrame({"row": np.arange(12), "is_click": [1, 1] + [0] * 10})
    first = sample_logged_negatives(frame, "is_click", negative_ratio=2, seed=7)
    second = sample_logged_negatives(frame, "is_click", negative_ratio=2, seed=7)
    assert len(first) == 6
    assert first["is_click"].sum() == 2
    assert first.equals(second)
    assert set(first["row"]).issubset(set(frame["row"]))


def test_pairwise_dataset_pairs_logged_positive_and_negative_per_user():
    features = np.arange(12).reshape(6, 2)
    labels = np.array([1, 0, 0, 1, 0, 1], dtype=np.float32)
    users = np.array([1, 1, 1, 2, 2, 3])
    base = InteractionDataset(features, labels, users, np.arange(6))
    pairs = PairwiseInteractionDataset(base, negatives_per_positive=2, seed=5)
    assert len(pairs) == 4
    for positive, negative in pairs:
        positive_index = int(torch.where((base.features == positive).all(dim=1))[0][0])
        negative_index = int(torch.where((base.features == negative).all(dim=1))[0][0])
        assert labels[positive_index] == 1
        assert labels[negative_index] == 0
        assert users[positive_index] == users[negative_index]
