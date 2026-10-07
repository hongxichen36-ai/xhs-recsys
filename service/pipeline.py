"""
推荐主流程
启动时构建召回器和排序器，请求时执行：召回 → 排序 → MMR
支持冷启动用户（行为 < 5 条）走兴趣类目 + 热门。
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
from recall.cat_recall import CatRecall
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

        # 加载用户偏好（冷启动 + 类目召回用）
        users_df = pd.read_csv(data_dir / "users.csv")
        self.user_prefs = {}
        for uid, p in zip(users_df["user_id"], users_df["pref_cats"]):
            if isinstance(p, str) and p:
                self.user_prefs[int(uid)] = p.split(",")
            else:
                self.user_prefs[int(uid)] = []

        # 划分：前 70% 训练召回器+特征，后 30% 训 LightGBM
        actions_sorted = self.actions.sort_values("timestamp")
        split1 = int(len(actions_sorted) * 0.7)
        self.recall_df = actions_sorted.iloc[:split1]
        self.rank_df = actions_sorted.iloc[split1:]

        # 构造用户历史
        print("[Pipeline] 构造用户历史...")
        self.user_history = {}
        for uid, g in self.actions.groupby("user_id"):
            self.user_history[int(uid)] = [
                {"note_id": int(n), "action": a}
                for n, a in zip(g["note_id"], g["action"])
            ]

        print("[Pipeline] 构建召回器...")
        self.itemcf = ItemCF(self.recall_df, top_k_sim=30)
        self.hot = HotRecall(self.recall_df, self.notes, top_k=1000)
        self.vector = ContentVectorRecall(self.notes)
        self.cat = CatRecall(self.recall_df, self.notes, per_cat=200)

        self.recall_manager = RecallManager(
            self.itemcf, self.hot,
            vector=self.vector,
            cat=self.cat,
        )

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
        users = [u for u in user_pos if u in self.user_history]

        # ✅ 从 300 提到 2000
        import random
        random.seed(42)
        random.shuffle(users)
        users = users[:2000]

        samples = []
        for uid in users:
            cands = self.recall_manager.recall(
                uid, k=200,       # ✅ 从 100 提到 200
                user_history=self.user_history[uid],
                pref_cats=self.user_prefs.get(uid, []),
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

        # ✅ 负采样：保留正样本，负样本降为正样本的 10 倍
        pos = rank_samples[rank_samples["label"] == 1]
        neg = rank_samples[rank_samples["label"] == 0]

        n_pos = len(pos)
        n_neg = min(len(neg), max(n_pos * 10, 5000))    # 至少 5000 负样本

        if len(neg) > n_neg:
            neg = neg.sample(n=n_neg, random_state=42)

        balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        print(f"[LightGBM] 采样后: 正 {len(pos)}, 负 {len(neg)}, "
            f"正负比 1:{len(neg) / max(len(pos), 1):.1f}")

        self.ranker = FineRanker(feature_cols=FeatureBuilder.feature_cols())
        self.ranker.train(balanced)

    def recommend(self, user_id, k=10):
        t0 = time.time()

        history = self.user_history.get(user_id, [])
        prefs = self.user_prefs.get(user_id, [])

        # 冷用户（行为 < 5 条）走兴趣类目 + 热门
        if len(history) < 5:
            items = self._cold_start_recall(prefs, k)
            return items, (time.time() - t0) * 1000

        candidates = self.recall_manager.recall(
            user_id, k=200,
            user_history=history,
            pref_cats=prefs,
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

    def _cold_start_recall(self, pref_cats, k=10):
        """冷启动召回：兴趣类目为主，热门兜底。"""
        results = []

        if pref_cats:
            per_cat = max(2, k // len(pref_cats))
            for cat in pref_cats:
                cat_items = [
                    n for n in self.hot.hot_items
                    if n in self.notes_idx.index
                    and self.notes_idx.loc[n, "category"] == cat
                ][:per_cat]
                for n in cat_items:
                    results.append({
                        "note_id": int(n),
                        "score": 0.9,
                        "source": "cold_interest",
                    })

        results += self.hot.recall(k=k)

        by_id = {}
        for r in results:
            nid = r["note_id"]
            if nid not in by_id or r["score"] > by_id[nid]["score"]:
                by_id[nid] = r

        return sorted(by_id.values(), key=lambda x: -x["score"])[:k]

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