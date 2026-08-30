from __future__ import annotations

import numpy as np


def ranking_metrics(
    users: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    ndcg_k: int = 10,
    recall_k: int = 50,
) -> dict[str, float]:
    """Macro-average ranking metrics over users with at least one positive."""
    users = np.asarray(users)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if not (len(users) == len(labels) == len(scores)):
        raise ValueError("users, labels, and scores must have equal lengths")
    if len(users) == 0:
        return {f"ndcg@{ndcg_k}": 0.0, f"recall@{recall_k}": 0.0, "evaluated_users": 0}

    # One global lexicographic sort avoids an O(users * rows) boolean scan.
    order = np.lexsort((-scores, users))
    sorted_users = users[order]
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    ndcgs: list[float] = []
    recalls: list[float] = []
    candidate_counts: list[int] = []
    for user_labels in np.split(sorted_labels, boundaries):
        positive_count = int(user_labels.sum())
        if positive_count == 0:
            continue
        gains = user_labels[:ndcg_k]
        discounts = 1.0 / np.log2(np.arange(len(gains)) + 2.0)
        dcg = float(np.sum(gains * discounts))
        ideal_length = min(positive_count, ndcg_k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(ideal_length) + 2.0)))
        ndcgs.append(dcg / idcg)
        recalls.append(float(user_labels[:recall_k].sum()) / positive_count)
        candidate_counts.append(len(user_labels))
    return {
        f"ndcg@{ndcg_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{recall_k}": float(np.mean(recalls)) if recalls else 0.0,
        "evaluated_users": len(ndcgs),
        "median_candidates_per_user": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "p95_candidates_per_user": float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0,
        "users_at_or_below_recall_k_fraction": (
            float(np.mean(np.asarray(candidate_counts) <= recall_k)) if candidate_counts else 0.0
        ),
    }
