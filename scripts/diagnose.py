"""
诊断：看召回质量到底如何
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
from collections import Counter

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


if __name__ == "__main__":
    data_dir = ROOT / "data" / "raw"
    actions = pd.read_csv(data_dir / "actions.csv")
    notes = pd.read_csv(data_dir / "notes.csv")
    users = pd.read_csv(data_dir / "users.csv")

    train, test = split_by_user(actions, test_ratio=0.2)
    print(f"训练集: {len(train)}, 测试集: {len(test)}")

    user_history = {}
    for uid, g in train.groupby("user_id"):
        user_history[int(uid)] = [
            {"note_id": int(n), "action": a}
            for n, a in zip(g["note_id"], g["action"])
        ]

    print("\n构建召回器...")
    itemcf = ItemCF(train, top_k_sim=30)
    hot = HotRecall(train, notes, top_k=1000)
    vector = ContentVectorRecall(notes)
    rm = RecallManager(itemcf, hot, vector=vector)

    note_cat = notes.set_index("note_id")["category"].to_dict()
    user_prefs = {
        int(uid): str(p).split(",") if isinstance(p, str) else []
        for uid, p in zip(users["user_id"], users["pref_cats"])
    }

    # === 诊断 1：看用户 0 的召回 ===
    uid = 0
    hist = user_history.get(uid, [])
    print(f"\n{'='*50}")
    print(f"用户 {uid}")
    print(f"历史条数: {len(hist)}")
    print(f"偏好类目: {user_prefs.get(uid)}")

    recs = rm.recall(uid, k=100, user_history=hist)
    print(f"召回条数: {len(recs)}")

    rec_cats = Counter(note_cat.get(r["note_id"]) for r in recs)
    rec_srcs = Counter(r["source"] for r in recs)
    print(f"召回来源分布: {dict(rec_srcs)}")
    print(f"召回类目分布: {dict(rec_cats)}")

    # === 诊断 2：命中率 ===
    hit_in_recall = sum(1 for r in recs if note_cat.get(r["note_id"]) in user_prefs.get(uid, []))
    print(f"命中偏好类目的召回条数: {hit_in_recall}/{len(recs)}")

    # === 诊断 3：如果扩大 k ===
    print(f"\n{'='*50}")
    print("扩大 k 后的 Recall")
    gt_map = test.groupby("user_id")["note_id"].apply(set).to_dict()

    for k in [100, 300, 500, 1000]:
        recalls = []
        for u in list(gt_map.keys())[:500]:
            if u not in user_history or len(user_history[u]) < 5:
                continue
            recs = rm.recall(u, k=k, user_history=user_history[u])
            rec_ids = {r["note_id"] for r in recs}
            gt = gt_map[u]
            if gt:
                recalls.append(len(rec_ids & gt) / len(gt))
        print(f"  Recall@{k}: {np.mean(recalls):.4f}")