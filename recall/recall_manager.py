"""
多路召回统一入口
ItemCF + 热门 + (可选) 向量召回
用 RRF (Reciprocal Rank Fusion) 融合，对分数尺度不敏感。
"""


class RecallManager:
    def __init__(self, itemcf, hot, vector=None,
                 weights=None, rrf_k=60):
        self.itemcf = itemcf
        self.hot = hot
        self.vector = vector
        self.rrf_k = rrf_k
        # 各路权重，作用在 RRF 分数上
        self.weights = weights or {
            "itemcf": 1.0,
            "hot": 0.3,
            "vector": 0.5,     # ← 向量权重降到 0.5
        }

    def recall(self, user_id, k=200, user_history=None):
        # 1. 收集各路召回的候选列表（按分数降序，即按排名）
        source_lists = {}

        itemcf_items = self.itemcf.recall(user_id, k=150)
        if itemcf_items:
            source_lists["itemcf"] = itemcf_items

        hot_items = self.hot.recall(k=50)
        if hot_items:
            source_lists["hot"] = hot_items

        if self.vector and user_history:
            history_ids = {h["note_id"] for h in user_history}
            vec_items = self.vector.recall(
                user_history, k=150, exclude_items=history_ids
            )
            if vec_items:
                source_lists["vector"] = vec_items

        # 2. RRF 融合
        rrf_scores = {}
        rrf_source = {}
        for source, items in source_lists.items():
            w = self.weights.get(source, 1.0)
            for rank, item in enumerate(items, start=1):
                nid = item["note_id"]
                add = w / (self.rrf_k + rank)
                rrf_scores[nid] = rrf_scores.get(nid, 0.0) + add
                # 记录贡献最大的来源，用于展示
                if nid not in rrf_source or add > rrf_source[nid][1]:
                    rrf_source[nid] = (source, add)

        # 3. 按 RRF 分数排序
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]

        return [
            {
                "note_id": nid,
                "score": score,
                "source": rrf_source[nid][0],
            }
            for nid, score in ranked
        ]