"""
离线评估
Recall@K / HitRate@K / 覆盖率
"""
import numpy as np
from collections import defaultdict


def split_train_test(actions_df, test_ratio=0.2):
    """
    按时间给每个用户划分训练/测试。
    最后 test_ratio 比例的行为作为测试。
    """
    train_list, test_list = [], []
    for uid, group in actions_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])
    import pandas as pd
    return pd.concat(train_list), pd.concat(test_list)


def recall_at_k(recommended, ground_truth, k=100):
    rec = set(recommended[:k])
    gt = set(ground_truth)
    if not gt:
        return 0.0
    return len(rec & gt) / len(gt)


def evaluate_recall(recall_manager, train_df, test_df,
                    n_users=200, k=100, verbose=True):
    """
    对每个用户：
      用 train_df 构造历史，用 test_df 的行为作为 ground truth，
      看召回结果是否命中。
    """
    # 训练集里每个用户的历史（用于向量召回）
    user_history = defaultdict(list)
    for _, r in train_df.iterrows():
        user_history[r["user_id"]].append(
            {"note_id": int(r["note_id"]), "action": r["action"]}
        )

    # 测试集里每个用户的正样本集合
    gt_map = defaultdict(set)
    for _, r in test_df.iterrows():
        gt_map[r["user_id"]].add(int(r["note_id"]))

    # 只评测训练集里出现过的用户（冷启动用户单独处理）
    valid_users = [u for u in gt_map if u in user_history]
    valid_users = valid_users[:n_users]

    recalls = []
    all_recalled = set()

    for uid in valid_users:
        recs = recall_manager.recall(
            uid, k=k, user_history=user_history[uid]
        )
        rec_ids = [r["note_id"] for r in recs]
        all_recalled.update(rec_ids)

        r = recall_at_k(rec_ids, gt_map[uid], k=k)
        recalls.append(r)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    coverage = len(all_recalled) / 5000.0   # 5000 篇笔记

    if verbose:
        print(f"\n📈 离线召回评估 (n_users={len(valid_users)}, k={k})")
        print(f"  Recall@{k}:  {mean_recall:.4f}")
        print(f"  覆盖率:      {coverage:.4f}  ({len(all_recalled)}/5000)")

    return {"recall": mean_recall, "coverage": coverage}