"""
排序评估：NDCG@K, HitRate@K
"""
import numpy as np


def ndcg_at_k(ranked_items, ground_truth, k=10):
    """ranked_items 已按分数降序排列"""
    dcg = 0.0
    for i, nid in enumerate(ranked_items[:k]):
        if nid in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked_items, ground_truth, k=10):
    return 1.0 if set(ranked_items[:k]) & ground_truth else 0.0