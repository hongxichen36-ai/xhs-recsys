"""
Day 4：MMR 多样性重排 + 冷启动 + AB 模拟
对比：
  - LightGBM 直排
  - LightGBM + MMR 重排
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
from rank.features import FeatureBuilder
from rank.fine_rank import FineRanker
from rerank.mmr import MMRReranker
from rerank.cold_start import ColdStartHandler
from evaluation.ranking_metrics import ndcg_at_k, hit_rate_at_k
from evaluation.diversity import (
    category_entropy, author_repetition_rate, intra_list_similarity
)


def split_by_user(actions_df, test_ratio=0.2):
    train_list, test_list = [], []
    for uid, group in actions_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])
    return pd.concat(train_list), pd.concat(test_list)


def build_training_samples(recall_manager, rank_train, recall_train,
                           feature_builder, n_users=300, k=100):
    user_pos = rank_train.groupby("user_id")["note_id"].apply(set).to_dict()
    user_hist = {}
    for uid, g in recall_train.groupby("user_id"):
        user_hist[uid] = [{"note_id": int(n), "action": a}
                          for n, a in zip(g["note_id"], g["action"])]

    users = [u for u in user_pos if u in user_hist][:n_users]
    print(f"[Sample] 采样 {len(users)} 个用户的排序训练样本")

    all_samples = []
    for uid in users:
        candidates = recall_manager.recall(
            uid, k=k, user_history=user_hist[uid]
        )
        if not candidates:
            continue
        feats = feature_builder.build(uid, candidates)
        feats["label"] = feats["note_id"].isin(user_pos[uid]).astype(int)
        all_samples.append(feats)
    return pd.concat(all_samples, ignore_index=True)


def build_user_hist(train_df):
    hist = {}
    for uid, g in train_df.groupby("user_id"):
        hist[uid] = [{"note_id": int(n), "action": a}
                     for n, a in zip(g["note_id"], g["action"])]
    return hist


def evaluate(recall_manager, ranker, feature_builder, mmr_reranker,
             test_df, train_df, notes_df,
             k_recall=200, k_eval=10, n_users=200, use_mmr=False):
    user_gt = test_df.groupby("user_id")["note_id"].apply(set).to_dict()
    user_hist = build_user_hist(train_df)
    valid_users = [u for u in user_gt if u in user_hist][:n_users]

    note_cat = notes_df.set_index("note_id")["category"].to_dict()
    note_author = notes_df.set_index("note_id")["author_id"].to_dict()

    ndcg_l, hit_l = [], []
    ent_l, ar_l, sim_l = [], [], []

    for uid in valid_users:
        candidates = recall_manager.recall(
            uid, k=k_recall, user_history=user_hist[uid]
        )
        if not candidates:
            continue
        feats = feature_builder.build(uid, candidates)
        if feats.empty:
            continue
        feats = feats.copy()
        feats["lgb_score"] = ranker.predict(feats)
        ranked = feats.sort_values("lgb_score", ascending=False)
        scored_items = ranked[["note_id", "lgb_score"]].rename(
            columns={"lgb_score": "score"}
        ).to_dict("records")

        if use_mmr:
            reranked = mmr_reranker.rerank(scored_items, k=k_eval)
            final_ids = [r["note_id"] for r in reranked]
        else:
            final_ids = [r["note_id"] for r in scored_items[:k_eval]]

        gt = user_gt[uid]
        ndcg_l.append(ndcg_at_k(final_ids, gt, k=k_eval))
        hit_l.append(hit_rate_at_k(final_ids, gt, k=k_eval))
        ent_l.append(category_entropy(final_ids, note_cat))
        ar_l.append(author_repetition_rate(final_ids, note_author))
        sim_l.append(intra_list_similarity(final_ids, note_cat))

    return {
        "ndcg": float(np.mean(ndcg_l)),
        "hit": float(np.mean(hit_l)),
        "entropy": float(np.mean(ent_l)),
        "author_rep": float(np.mean(ar_l)),
        "intra_sim": float(np.mean(sim_l)),
    }


if __name__ == "__main__":
    data_dir = ROOT / "data" / "raw"
    actions = pd.read_csv(data_dir / "actions.csv")
    notes = pd.read_csv(data_dir / "notes.csv")

    train, test = split_by_user(actions, test_ratio=0.2)
    print(f"训练集: {len(train)}, 测试集: {len(test)}")

    train_sorted = train.sort_values("timestamp")
    split = int(len(train_sorted) * 0.7)
    recall_train = train_sorted.iloc[:split]
    rank_train = train_sorted.iloc[split:]

    print("\n[1/4] 构建召回器...")
    itemcf = ItemCF(recall_train, top_k_sim=50)
    hot = HotRecall(recall_train, notes, top_k=500)
    vector = ContentVectorRecall(notes)
    recall_manager = RecallManager(itemcf, hot, vector=vector)

    print("[2/4] 构建特征统计...")
    feature_builder = FeatureBuilder(recall_train, notes)

    print("[3/4] 训练 LightGBM...")
    rank_samples = build_training_samples(
        recall_manager, rank_train, recall_train,
        feature_builder, n_users=300, k=100,
    )
    ranker = FineRanker(feature_cols=FeatureBuilder.feature_cols())
    ranker.train(rank_samples)

    print("[4/4] 初始化 MMR 重排器...")
    mmr = MMRReranker(notes, note_emb=vector.note_emb, lambda_=0.7)

    # ========== 对比 1：LGB 直排 vs LGB + MMR ==========
    print("\n" + "=" * 60)
    print("评估 1：LightGBM 直排 vs LightGBM + MMR")
    print("=" * 60)
    res_base = evaluate(recall_manager, ranker, feature_builder, mmr,
                        test, train, notes, k_eval=10,
                        n_users=200, use_mmr=False)
    res_mmr = evaluate(recall_manager, ranker, feature_builder, mmr,
                       test, train, notes, k_eval=10,
                       n_users=200, use_mmr=True)

    print(f"\n{'指标':<15}{'LGB':>10}{'LGB+MMR':>12}{'变化':>12}")
    for key in ["ndcg", "hit", "entropy", "author_rep", "intra_sim"]:
        b, m = res_base[key], res_mmr[key]
        delta = (m - b) / max(abs(b), 1e-9) * 100
        print(f"{key:<15}{b:>10.4f}{m:>12.4f}{delta:>+11.1f}%")

    # ========== 对比 2：冷启动 ==========
    print("\n" + "=" * 60)
    print("评估 2：冷启动策略")
    print("=" * 60)
    cold = ColdStartHandler(hot, notes, min_user_actions=5)

    user_counts = recall_train.groupby("user_id").size()
    cold_users = user_counts[user_counts < 5].index.tolist()
    print(f"冷用户数 (<5 行为): {len(cold_users)}")

    if not cold_users:
        print("  当前数据没有冷用户，模拟新用户（无历史）:")
        recs = cold.recall_for_cold_user(k=10)
        note_cat = notes.set_index("note_id")["category"].to_dict()
        cats = [note_cat[r["note_id"]] for r in recs]
        print(f"    top10 类目: {cats}")
    else:
        for uid in cold_users[:3]:
            g = recall_train[recall_train["user_id"] == uid]
            hist = [{"note_id": int(n), "action": a}
                    for n, a in zip(g["note_id"], g["action"])]
            recs = cold.recall_for_cold_user(k=10)
            note_cat = notes.set_index("note_id")["category"].to_dict()
            cats = [note_cat[r["note_id"]] for r in recs]
            print(f"  用户 {uid}: history={len(hist)}, top10类目={cats}")