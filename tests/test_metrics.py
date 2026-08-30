import numpy as np

from research_rec.metrics import ranking_metrics


def test_ranking_metrics_are_macro_averaged_per_user():
    users = np.array([1, 1, 1, 2, 2])
    labels = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.9, 0.1])
    result = ranking_metrics(users, labels, scores, ndcg_k=1, recall_k=1)
    assert result["ndcg@1"] == 0.5
    assert result["recall@1"] == 0.25
    assert result["evaluated_users"] == 2


def test_users_without_positives_are_excluded():
    result = ranking_metrics(np.array([1, 1]), np.array([0, 0]), np.array([0.2, 0.1]))
    assert result["evaluated_users"] == 0
    assert result["ndcg@10"] == 0.0

