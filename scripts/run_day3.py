import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
"""
Day 3：特征工程 + LightGBM 排序
对比：RRF 直接排序 vs LightGBM 排序
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from collections import defaultdict

from recall.itemcf import ItemCF
from recall.hot import HotRecall
from recall.two_tower import ContentVectorRecall
from recall.recall_manager import RecallManager
from rank.features import FeatureBuilder
from rank.fine_rank import FineRanker
from evaluation.ranking_metrics import ndcg_at_k, hit_rate_at_k


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
    """
    对每个用户：
      1. 用 recall_train 里的用户历史做召回
      2. label = 候选 note 是否出现在 rank_train（后 30%）里
    """
    user_pos = rank_train.groupby("user_id")["note_id"].apply(set).to_dict()
    user_hist = recall_train.groupby("user_id").apply(
        lambda g: [{"note_id": int(n), "action": a}
                   for n, a in zip(g["note_id"], g["action"])]
    ).to_dict()

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


def evaluate_ranking(recall_manager, ranker, feature_builder,
                     test_df, train_df, k_recall=200, k_eval=10, n_users=200):
    """对比：RRF 直排 vs LightGBM 排序"""
    user_gt = test_df.groupby("user_id")["note_id"].apply(set).to_dict()
    user_hist = train_df.groupby("user_id").apply(
        lambda g: [{"note_id": int(n), "action": a}
                   for n, a in zip(g["note_id"], g["action"])]
    ).to_dict()

    valid_users = [u for u in user_gt if u in user_hist][:n_users]

    rrf_ndcg, rrf_hit = [], []
    lgb_ndcg, lgb_hit = [], []

    for uid in valid_users:
        candidates = recall_manager.recall(
            uid, k=k_recall, user_history=user_hist[uid]
        )
        if not candidates:
            continue
        feats = feature_builder.build(uid, candidates)
        if feats.empty:
            continue

        gt = user_gt[uid]

        # 基线：RRF 直接排序
        rrf_ranked = feats.sort_values("rrf_score", ascending=False)["note_id"].tolist()
        rrf_ndcg.append(ndcg_at_k(rrf_ranked, gt, k=k_eval))
        rrf_hit.append(hit_rate_at_k(rrf_ranked, gt, k=k_eval))

        # LightGBM 排序
        feats = feats.copy()
        feats["lgb_score"] = ranker.predict(feats)
        lgb_ranked = feats.sort_values("lgb_score", ascending=False)["note_id"].tolist()
        lgb_ndcg.append(ndcg_at_k(lgb_ranked, gt, k=k_eval))
        lgb_hit.append(hit_rate_at_k(lgb_ranked, gt, k=k_eval))

    return {
        "rrf_ndcg@10": float(np.mean(rrf_ndcg)),
        "rrf_hit@10": float(np.mean(rrf_hit)),
        "lgb_ndcg@10": float(np.mean(lgb_ndcg)),
        "lgb_hit@10": float(np.mean(lgb_hit)),
    }


if __name__ == "__main__":
    data_dir = ROOT / "data" / "raw"
    actions = pd.read_csv(data_dir / "actions.csv")
    notes = pd.read_csv(data_dir / "notes.csv")

    # === 1. 划分 ===
    train, test = split_by_user(actions, test_ratio=0.2)
    print(f"训练集: {len(train)}, 测试集: {len(test)}")

    # train 再切：前 70% 训召回器/特征，后 30% 作为排序 label
    train_sorted = train.sort_values("timestamp")
    split = int(len(train_sorted) * 0.7)
    recall_train = train_sorted.iloc[:split]
    rank_train = train_sorted.iloc[split:]
    print(f"召回训练: {len(recall_train)}, 排序训练: {len(rank_train)}")

    # === 2. 用 recall_train 训练召回器 + 特征统计 ===
    print("\n[1/4] 构建召回器...")
    itemcf = ItemCF(recall_train, top_k_sim=50)
    hot = HotRecall(recall_train, notes, top_k=500)
    vector = ContentVectorRecall(notes)
    recall_manager = RecallManager(itemcf, hot, vector=vector)

    print("\n[2/4] 构建特征统计...")
    feature_builder = FeatureBuilder(recall_train, notes)

    # === 3. 构造排序训练样本 ===
    print("\n[3/4] 构造排序训练样本...")
    rank_samples = build_training_samples(
        recall_manager, rank_train, recall_train,
        feature_builder, n_users=300, k=100,
    )
    print(f"排序训练样本: {len(rank_samples)}, "
          f"正样本: {rank_samples['label'].sum()}")

    # === 4. 训练 LightGBM ===
    print("\n[4/4] 训练 LightGBM...")
    ranker = FineRanker(feature_cols=FeatureBuilder.feature_cols())
    ranker.train(rank_samples)

    # === 5. 评估：RRF 直排 vs LightGBM ===
    print("\n" + "=" * 50)
    print("评估：RRF 直排 vs LightGBM 排序")
    print("=" * 50)
    results = evaluate_ranking(
        recall_manager, ranker, feature_builder,
        test_df=test, train_df=train,
        k_recall=200, k_eval=10, n_users=200,
    )
    print(f"  RRF  NDCG@10 = {results['rrf_ndcg@10']:.4f}, "
          f"Hit@10 = {results['rrf_hit@10']:.4f}")
    print(f"  LGB  NDCG@10 = {results['lgb_ndcg@10']:.4f}, "
          f"Hit@10 = {results['lgb_hit@10']:.4f}")

    delta = (results['lgb_ndcg@10'] - results['rrf_ndcg@10']) / \
            max(results['rrf_ndcg@10'], 1e-9) * 100
    print(f"\n  NDCG@10 提升: {delta:+.1f}%")

    # === 6. 特征重要性 ===
    print("\n📊 特征重要性 Top-10:")
    imp = ranker.feature_importance().head(10)
    for _, row in imp.iterrows():
        print(f"  {row['feature']:20s}  {row['importance']:.0f}")