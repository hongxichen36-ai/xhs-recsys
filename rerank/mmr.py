"""
MMR (Maximal Marginal Relevance) 多样性重排
给定已按相关性排序的候选列表，重新排序，让列表既相关又多样。
"""
import numpy as np


class MMRReranker:
    def __init__(self, notes_df, note_emb=None, lambda_=0.7):
        """
        lambda_: 0~1，越大越偏相关性，越小越偏多样性
        note_emb: (N_ITEM, dim) 的 embedding 矩阵，可选
        """
        self.lambda_ = lambda_
        self.note_cat = notes_df.set_index("note_id")["category"].to_dict()
        self.note_author = notes_df.set_index("note_id")["author_id"].to_dict()
        self.note_emb = note_emb
        self.nid2row = None
        if note_emb is not None:
            self.note_ids = notes_df["note_id"].values
            self.nid2row = {int(n): i for i, n in enumerate(self.note_ids)}

    def _sim(self, a, b):
        """两个 note 的相似度，0~1"""
        if a == b:
            return 1.0
        sim = 0.0
        if self.note_author.get(a) == self.note_author.get(b):
            sim += 0.4
        if self.note_cat.get(a) == self.note_cat.get(b):
            sim += 0.3
        if self.note_emb is not None and self.nid2row is not None:
            ra, rb = self.nid2row.get(a), self.nid2row.get(b)
            if ra is not None and rb is not None:
                cos = float(np.dot(self.note_emb[ra], self.note_emb[rb]))
                sim += 0.3 * max(cos, 0.0)
        return min(sim, 1.0)

    def rerank(self, scored_items, k=10):
        """
        scored_items: [{"note_id":..., "score":...}, ...] 已按 score 降序
        返回: 重排后的 top-k
        """
        if not scored_items:
            return []

        scores = [r["score"] for r in scored_items]
        smax, smin = max(scores), min(scores)
        if smax - smin < 1e-9:
            relevance = {r["note_id"]: 1.0 for r in scored_items}
        else:
            relevance = {r["note_id"]: (r["score"] - smin) / (smax - smin)
                         for r in scored_items}

        selected = []
        remaining = list(scored_items)

        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda x: relevance[x["note_id"]])
            else:
                selected_ids = [s["note_id"] for s in selected]

                def mmr_score(item):
                    nid = item["note_id"]
                    rel = relevance[nid]
                    max_sim = max(self._sim(nid, sid) for sid in selected_ids)
                    return self.lambda_ * rel - (1 - self.lambda_) * max_sim

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining = [r for r in remaining
                         if r["note_id"] != best["note_id"]]

        return selected