"""
多路召回统一入口
ItemCF + Hot + Vector + Cat
用 RRF 融合，对分数尺度不敏感。
"""


class RecallManager:
    def __init__(self, itemcf, hot, vector=None, cat=None, weights=None):
        self.itemcf = itemcf
        self.hot = hot
        self.vector = vector
        self.cat = cat
        self.weights = weights or {
            "itemcf": 1.0,
            "hot": 0.3,
            "vector": 1.0,
            "cat": 0.6,      # ← 类目召回权重
        }

    def recall(self, user_id, k=200, user_history=None, pref_cats=None):
        """
        user_history: [{note_id, action}, ...] 用于向量召回
        pref_cats:    ['旅行', '数码']         用于类目召回
        """
        source_lists = {}

        # 1. ItemCF
        itemcf_items = self.itemcf.recall(user_id, k=150)
        if itemcf_items:
            source_lists["itemcf"] = itemcf_items

        # 2. Hot
        hot_items = self.hot.recall(k=50)
        if hot_items:
            source_lists["hot"] = hot_items

        # 3. Vector
        if self.vector and user_history:
            history_ids = {h["note_id"] for h in user_history}
            vec_items = self.vector.recall(
                user_history, k=150, exclude_items=history_ids
            )
            if vec_items:
                source_lists["vector"] = vec_items

        # 4. Cat（新增）
        if self.cat and pref_cats:
            cat_items = self.cat.recall(user_id, pref_cats, k=150)
            if cat_items:
                source_lists["cat"] = cat_items

        # RRF 融合
        rrf_scores = {}
        rrf_source = {}
        for source, items in source_lists.items():
            w = self.weights.get(source, 1.0)
            for rank, item in enumerate(items, start=1):
                nid = item["note_id"]
                add = w / (60 + rank)
                rrf_scores[nid] = rrf_scores.get(nid, 0.0) + add
                if nid not in rrf_source or add > rrf_source[nid][1]:
                    rrf_source[nid] = (source, add)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        return [
            {"note_id": nid, "score": s, "source": rrf_source[nid][0]}
            for nid, s in ranked
        ]