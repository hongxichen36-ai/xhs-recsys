"""
热门召回
全局热门 + 类目热门。
"""
import pandas as pd


class HotRecall:
    def __init__(self, actions_df, notes_df, top_k=500):
        self.top_k = top_k
        w = {"click": 1.0, "like": 2.0, "collect": 3.0, "comment": 4.0}

        # 加权热度
        actions_df = actions_df.copy()
        actions_df["w"] = actions_df["action"].map(w)
        counts = actions_df.groupby("note_id")["w"].sum().sort_values(ascending=False)

        self.hot_items = counts.head(top_k).index.tolist()
        self.hot_scores = (counts.head(top_k) / counts.max()).to_dict()

        # 类目热门
        note_cat = notes_df.set_index("note_id")["category"].to_dict()
        self.cat_of = note_cat
        self.cat_hot = {}
        for cat in notes_df["category"].unique():
            ids = [n for n in self.hot_items if note_cat.get(n) == cat]
            self.cat_hot[cat] = ids[:50]

        print(f"[HotRecall] {len(self.hot_items)} hot items, "
              f"{len(self.cat_hot)} categories")

    def recall(self, k=50, category=None):
        if category and category in self.cat_hot:
            items = self.cat_hot[category][:k]
        else:
            items = self.hot_items[:k]

        return [{"note_id": int(n),
                 "score": float(self.hot_scores.get(n, 0.5)),
                 "source": "hot"}
                for n in items]