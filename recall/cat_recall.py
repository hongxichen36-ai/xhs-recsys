"""
类目召回
对用户偏好类目，取该类目里热度最高的笔记。
覆盖 ItemCF/Vector 漏掉的"热门新内容"。
"""
import numpy as np
import pandas as pd
from collections import defaultdict


ACTION_W = {"click": 1.0, "like": 2.0, "collect": 3.0, "comment": 4.0}


class CatRecall:
    def __init__(self, actions_df, notes_df, per_cat=100):
        """
        per_cat: 每个类目保留多少篇热门笔记
        """
        self.per_cat = per_cat
        self.note_cat = notes_df.set_index("note_id")["category"].to_dict()

        # 1. 加权热度
        tmp = actions_df.copy()
        tmp["w"] = tmp["action"].map(ACTION_W)
        note_pop = tmp.groupby("note_id")["w"].sum().to_dict()

        # 2. 按类目分组，每组按热度降序
        cat_items = defaultdict(list)
        for nid, cat in self.note_cat.items():
            cat_items[cat].append((nid, note_pop.get(nid, 0.0)))

        self.cat_top = {}
        for cat, items in cat_items.items():
            items.sort(key=lambda x: -x[1])
            self.cat_top[cat] = [nid for nid, _ in items[:per_cat]]

        # 3. 归一化分数（用于融合）
        self.note_pop = note_pop
        self.max_pop = max(note_pop.values()) if note_pop else 1.0

        print(f"[CatRecall] {len(self.cat_top)} 类目, "
              f"每类目 top-{per_cat}")

    def recall(self, user_id, pref_cats, k=100):
        """
        pref_cats: 用户的偏好类目列表，如 ['旅行', '数码']
        """
        if not pref_cats:
            return []

        results = []
        per_cat = max(5, k // len(pref_cats))

        for cat in pref_cats:
            items = self.cat_top.get(cat, [])[:per_cat]
            for nid in items:
                pop = self.note_pop.get(nid, 0.0) / self.max_pop
                results.append({
                    "note_id": int(nid),
                    "score": float(pop),
                    "source": "cat",
                })
        return results