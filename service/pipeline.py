"""
推荐主流程
启动时构建召回器和排序器，请求时执行：召回 → 排序 → MMR
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import pandas as pd
from collections import defaultdict

from recall.itemcf import ItemCF
from recall.hot import HotRecall
from recall.two_tower import ContentVectorRecall
from recall.recall_manager import RecallManager
from rank.features import FeatureBuilder
from rank.fine_rank import FineRanker
from rerank.mmr import MMRReranker


class RecommendPipeline:
    def __init__(self, data_dir=None):
        data_dir = Path(data_dir) if data_dir else ROOT / "data" / "raw"
        print("[Pipeline] 加载数据...")
        self.actions = pd.read_csv(data_dir / "actions.csv")
        self.notes = pd.read_csv(data_dir / "notes.csv")
        self.notes_idx = self.notes.set_index("note_id")

        # 划分：前 70% 训练召回器+特征，后 30% 训 LightGBM
        actions_sorted = self.actions.sort_values("timestamp")
        split1 = int(len(actions_sorted) * 0.7)
        self.recall_df = actions_sorted.iloc[:split1]
        self.rank_df = actions_sorted.iloc[split1:]

        # 先构造 user_history（要在 _train_ranker 之前）
        print("[Pipeline] 构造用户历史...")
        self.user_history = {}
        for uid, g in self.actions.groupby("user_id"):
            self.user_history[int(uid)] = [
                {"note_id": int(n), "action": a}
                for n, a in zip(g["note_id"], g["action"])
            ]

        print("[Pipeline] 构建召回器...")
        self.itemcf = ItemCF(self.recall_df, top_k_sim=50)
        self.hot = HotRecall(self.recall_df, self.notes, top_k=500)
        self.vector = ContentVectorRecall(self.notes)
        self.recall_manager = RecallManager(self.itemcf, self.hot, vector=self.vector)

        print("[Pipeline] 构建特征统计...")
        self.feature_builder = FeatureBuilder(self.recall_df, self.notes)

        print("[Pipeline] 训练 LightGBM...")
        self.ranker = None
        self._train_ranker()

        print("[Pipeline] 初始化 MMR...")
        self.mmr = MMRReranker(self.notes, note_emb=self.vector.note_emb, lambda_=0.7)

        print("[Pipeline] 就绪 ✅")

    def _train_ranker(self):
        user_pos = self.rank_df.groupby("user_id")["note_id"].apply(set).to_dict()
        users = [u for u in user_pos if u in self.user_history][:300]

        samples = []
        for uid in users:
            cands = self.recall_manager.recall(
                uid, k=100, user_history=self.user_history[uid]
            )
            if not cands:
                continue
            feats = self.feature_builder.build(uid, cands)
            feats["label"] = feats["note_id"].isin(user_pos[uid]).astype(int)
            samples.append(feats)

        if not samples:
            print("[Pipeline] ⚠️ 无法构造训练样本")
            return

        rank_samples = pd.concat(samples, ignore_index=True)
        self.ranker = FineRanker(feature_cols=FeatureBuilder.feature_cols())
        self.ranker.train(rank_samples)

    def recommend(self, user_id, k=10):
        t0 = time.time()

        history = self.user_history.get(user_id, [])
        if not history:
            # 冷启动：直接走热门
            items = self.hot.recall(k=k)
            return items, (time.time() - t0) * 1000

        candidates = self.recall_manager.recall(
            user_id, k=200, user_history=history
        )
        if not candidates:
            return [], (time.time() - t0) * 1000

        feats = self.feature_builder.build(user_id, candidates)
        if feats.empty:
            return [], (time.time() - t0) * 1000

        feats = feats.copy()
        feats["lgb_score"] = self.ranker.predict(feats)
        ranked = feats.sort_values("lgb_score", ascending=False)
        scored_items = ranked[["note_id", "lgb_score"]].rename(
            columns={"lgb_score": "score"}
        ).to_dict("records")

        reranked = self.mmr.rerank(scored_items, k=k)
        latency = (time.time() - t0) * 1000
        return reranked, latency

    def get_note_info(self, note_id):
        if note_id not in self.notes_idx.index:
            return {"note_id": note_id, "title": "未知", "category": "未知", "tags": ""}
        row = self.notes_idx.loc[note_id]
        return {
            "note_id": int(note_id),
            "title": str(row["title"]),
            "category": str(row["category"]),
            "tags": str(row["tags"]),
        }