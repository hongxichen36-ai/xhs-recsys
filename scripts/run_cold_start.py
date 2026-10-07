"""
冷启动评估：分热用户 / 冷用户两组
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from recall.itemcf import ItemCF
from recall.hot import HotRecall
from recall.two_tower import ContentVectorRecall
from recall.recall_manager import RecallManager


def split_by_user(actions_df, test_ratio=0.2):
    train_list, test_list = [], []
    for uid, group in actions_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])
    return pd.concat(train_list), pd.concat(test_list)


def evaluate_group(recall_manager, train_df, test_df, user_list,
                   user_history, k=100):
    gt_map = test_df.groupby("user_id")["note_id"].apply(set).to_dict()

    recalls = []
    for uid in user_list:
        if uid not in gt_map:
            continue
        recs = recall_manager.recall(uid, k=k, user_history=user_history.get(uid, []))
        rec_ids = {r["note_id"] for r in recs}
        gt = gt_map[uid]
        if gt:
            recalls.append(len(rec_ids & gt) / len(gt))
    return float(np.mean(recalls)) if recalls else 0.0


if __name__ == "__main__":
    data_dir = ROOT / "data" / "raw"
    actions = pd.read_csv(data_dir / "actions.csv")
    notes = pd.read_csv(data_dir / "notes.csv")
    users = pd.read_csv(data_dir / "users.csv")

    train, test = split_by_user(actions, test_ratio=0.2)
    print(f"训练集: {len(train)}, 测试集: {len(test)}")

    # 用户历史
    user_history = {}
    for uid, g in train.groupby("user_id"):
        user_history[int(uid)] = [
            {"note_id": int(n), "action": a}
            for n, a in zip(g["note_id"], g["action"])
        ]

    # 构建召回
    print("\n构建召回器...")
    itemcf = ItemCF(train, top_k_sim=30)
    hot = HotRecall(train, notes, top_k=1000)
    vector = ContentVectorRecall(notes)
    rm = RecallManager(itemcf, hot, vector=vector)

    # 分两组：热用户 / 冷用户
    users_with_history = train.groupby("user_id").size()
    hot_users = users_with_history[users_with_history >= 5].index.tolist()
    cold_users = users_with_history[users_with_history < 5].index.tolist()

    print(f"\n热用户数: {len(hot_users)}, 冷用户数: {len(cold_users)}")

    # 评估
    print("\n" + "=" * 50)
    print("冷启动评估")
    print("=" * 50)

    hot_recall = evaluate_group(rm, train, test, hot_users[:500],
                                 user_history, k=100)
    cold_recall = evaluate_group(rm, train, test, cold_users[:200],
                                  user_history, k=100)

    print(f"  热用户（≥5 行为）Recall@100: {hot_recall:.4f}")
    print(f"  冷用户（<5 行为）Recall@100: {cold_recall:.4f}")

    # 冷用户直接走热门
    print("\n对比：冷用户走纯热门召回")
    def hot_only_recall_fn(uid, k=100, user_history=None):
        return hot.recall(k=k)

    # 简易评估
    gt_map = test.groupby("user_id")["note_id"].apply(set).to_dict()
    recalls = []
    for uid in cold_users[:200]:
        if uid not in gt_map:
            continue
        recs = hot.recall(k=100)
        rec_ids = {r["note_id"] for r in recs}
        gt = gt_map[uid]
        if gt:
            recalls.append(len(rec_ids & gt) / len(gt))
    hot_only = float(np.mean(recalls)) if recalls else 0.0
    print(f"  冷用户走热门兜底 Recall@100: {hot_only:.4f}")