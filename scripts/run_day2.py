"""
Day 2 对比：基础召回 vs 加向量召回
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from collections import defaultdict
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


def evaluate(recall_fn, train_df, test_df, notes_df, k=100, n_users=200):
    user_history = defaultdict(list)
    for _, r in train_df.iterrows():
        user_history[r["user_id"]].append(
            {"note_id": int(r["note_id"]), "action": r["action"]}
        )

    gt_map = defaultdict(set)
    for _, r in test_df.iterrows():
        gt_map[r["user_id"]].add(int(r["note_id"]))

    valid_users = [u for u in gt_map if u in user_history][:n_users]

    recalls, all_recalled = [], set()
    for uid in valid_users:
        recs = recall_fn(uid, k=k, user_history=user_history[uid])
        rec_ids = [r["note_id"] for r in recs]
        all_recalled.update(rec_ids)
        gt = gt_map[uid]
        if gt:
            recalls.append(len(set(rec_ids) & gt) / len(gt))

    return float(np.mean(recalls)), len(all_recalled) / len(notes_df)


if __name__ == "__main__":
    import numpy as np

    data_dir = ROOT / "data" / "raw"
    actions = pd.read_csv(data_dir / "actions.csv")
    notes = pd.read_csv(data_dir / "notes.csv")

    train, test = split_by_user(actions, test_ratio=0.2)
    print(f"训练集: {len(train)}, 测试集: {len(test)}")

    itemcf = ItemCF(train, top_k_sim=50)
    hot = HotRecall(train, notes, top_k=500)

    # ===== 基础：ItemCF + Hot =====
    print("\n" + "=" * 50)
    print("基础：ItemCF + Hot")
    print("=" * 50)
    rm_base = RecallManager(itemcf, hot)
    r_base, c_base = evaluate(rm_base.recall, train, test, notes, k=100)
    print(f"📊 Recall@100 = {r_base:.4f}")
    print(f"📊 Coverage   = {c_base:.4f}")

    # ===== 完整：+ 向量 =====
    print("\n" + "=" * 50)
    print("完整：ItemCF + Hot + Vector")
    print("=" * 50)
    vector = ContentVectorRecall(notes)
    rm_full = RecallManager(itemcf, hot, vector=vector)
    r_full, c_full = evaluate(rm_full.recall, train, test, notes, k=100)
    print(f"📊 Recall@100 = {r_full:.4f}")
    print(f"📊 Coverage   = {c_full:.4f}")

    # ===== 对比 =====
    print("\n" + "=" * 50)
    print("📈 对比")
    print("=" * 50)
    delta = (r_full - r_base) / max(r_base, 1e-9) * 100
    print(f"  Recall@100: {r_base:.4f} -> {r_full:.4f}  ({delta:+.1f}%)")
    print(f"  Coverage:   {c_base:.4f} -> {c_full:.4f}")