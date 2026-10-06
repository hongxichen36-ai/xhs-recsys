"""
特征工程
用户侧 + 笔记侧 + 交叉 + 召回分
"""
import numpy as np
import pandas as pd
from collections import defaultdict


ACTION_W = {"click": 1.0, "like": 2.0, "collect": 3.0, "comment": 4.0}


class FeatureBuilder:
    def __init__(self, train_df, notes_df):
        self.notes_df = notes_df
        self.note_cat = notes_df.set_index("note_id")["category"].to_dict()
        self.note_author = notes_df.set_index("note_id")["author_id"].to_dict()

        # 笔记热度（加权）
        tmp = train_df.copy()
        tmp["w"] = tmp["action"].map(ACTION_W)
        self.note_pop = tmp.groupby("note_id")["w"].sum().to_dict()
        self.note_max_pop = max(self.note_pop.values()) if self.note_pop else 1.0

        # 类目热度
        tmp["cat"] = tmp["note_id"].map(self.note_cat)
        self.cat_pop = tmp.groupby("cat").size().to_dict()
        self.cat_max_pop = max(self.cat_pop.values()) if self.cat_pop else 1.0

        # 用户行为数
        self.user_action_count = train_df.groupby("user_id").size().to_dict()

        # 用户-类目 计数
        self.user_cat_count = defaultdict(lambda: defaultdict(int))
        self.user_author_count = defaultdict(lambda: defaultdict(int))
        for uid, nid in zip(train_df["user_id"], train_df["note_id"]):
            cat = self.note_cat.get(nid, "unknown")
            au = self.note_author.get(nid, -1)
            self.user_cat_count[uid][cat] += 1
            self.user_author_count[uid][au] += 1

        # 用户 top-3 偏好类目集合
        self.user_pref_cats = {}
        for uid, d in self.user_cat_count.items():
            top3 = sorted(d.items(), key=lambda x: -x[1])[:3]
            self.user_pref_cats[uid] = {c for c, _ in top3}

    def build(self, uid, candidates):
        """candidates: [{"note_id":..., "source":..., "score":...}, ...]"""
        rows = []
        u_count = self.user_action_count.get(uid, 0)
        u_pref = self.user_pref_cats.get(uid, set())

        # 按来源归集分数
        src_scores = {"itemcf": {}, "hot": {}, "vector": {}}
        for c in candidates:
            s = c["source"]
            if s in src_scores:
                src_scores[s][c["note_id"]] = c["score"]

        for c in candidates:
            nid = c["note_id"]
            cat = self.note_cat.get(nid, "unknown")
            au = self.note_author.get(nid, -1)
            n_src = sum(1 for s in src_scores if nid in src_scores[s])

            rows.append({
                "user_id": uid,
                "note_id": nid,
                "user_action_count": u_count,
                "user_pref_match": int(cat in u_pref),
                "user_cat_count": self.user_cat_count[uid].get(cat, 0),
                "user_author_count": self.user_author_count[uid].get(au, 0),
                "note_pop": self.note_pop.get(nid, 0),
                "note_pop_norm": self.note_pop.get(nid, 0) / self.note_max_pop,
                "cat_pop_norm": self.cat_pop.get(cat, 0) / self.cat_max_pop,
                "itemcf_score": src_scores["itemcf"].get(nid, 0.0),
                "hot_score": src_scores["hot"].get(nid, 0.0),
                "vector_score": src_scores["vector"].get(nid, 0.0),
                "rrf_score": c["score"],
                "n_recall_sources": n_src,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def feature_cols():
        return [
            "user_action_count", "user_pref_match", "user_cat_count",
            "user_author_count", "note_pop", "note_pop_norm", "cat_pop_norm",
            "itemcf_score", "hot_score", "vector_score", "rrf_score",
            "n_recall_sources",
        ]