"""
ItemCF 召回
基于用户行为共现计算 item-item 相似度。
"""
import numpy as np
from collections import defaultdict


class ItemCF:
    def __init__(self, actions_df, top_k_sim=50):
        self.top_k_sim = top_k_sim
        self.item_sim = {}                       # item -> [(item, sim), ...]
        self.user_history = defaultdict(dict)    # user -> {item: weight}
        self._build(actions_df)

    def _build(self, actions_df):
        # 行为权重：收藏 > 评论 > 点赞 > 点击
        w = {"click": 1.0, "like": 2.0, "collect": 3.0, "comment": 4.0}

        # 1. user -> {item: weight}
        for uid, nid, act in zip(
            actions_df["user_id"].values,
            actions_df["note_id"].values,
            actions_df["action"].values,
        ):
            self.user_history[uid][nid] = \
                self.user_history[uid].get(nid, 0) + w[act]

        # 2. item -> 用户集合
        item_users = defaultdict(set)
        for u, items in self.user_history.items():
            for i in items:
                item_users[i].add(u)

        # 3. item 共现矩阵（稀疏 dict）
        co = defaultdict(lambda: defaultdict(float))
        for u, items in self.user_history.items():
            items_list = list(items.keys())
            for i in items_list:
                for j in items_list:
                    if i == j:
                        continue
                    co[i][j] += 1.0

        # 4. 归一化 + top-k
        for i, related in co.items():
            sims = sorted(related.items(), key=lambda x: -x[1])[:self.top_k_sim]
            total = sum(s for _, s in sims) or 1.0
            self.item_sim[i] = [(j, s / total) for j, s in sims]

        print(f"[ItemCF] {len(self.item_sim)} items with similarity")

    def recall(self, user_id, k=100):
        scores = defaultdict(float)
        history = self.user_history.get(user_id, {})
        for i, w in history.items():
            for j, s in self.item_sim.get(i, []):
                if j in history:
                    continue
                scores[j] += w * s

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [{"note_id": int(n), "score": float(s), "source": "itemcf"}
                for n, s in ranked]