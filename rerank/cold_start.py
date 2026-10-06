"""
冷启动策略
- 新用户：兴趣类目 + 热门兜底
- 新笔记：用内容相似度匹配（此处简化，可扩展）
"""
from collections import defaultdict


class ColdStartHandler:
    def __init__(self, hot_recall, notes_df, min_user_actions=5):
        self.hot = hot_recall
        self.notes_df = notes_df
        self.min_user_actions = min_user_actions
        self.note_cat = notes_df.set_index("note_id")["category"].to_dict()

        # 类目 -> 笔记列表（按热度排序，这里用出现顺序）
        self.cat_items = defaultdict(list)
        for nid, cat in self.note_cat.items():
            self.cat_items[cat].append(int(nid))

    def is_cold_user(self, user_history):
        """行为数少于阈值，视为冷用户"""
        return len(user_history) < self.min_user_actions

    def recall_for_cold_user(self, k=100, interest_cats=None):
        """
        interest_cats: 用户注册时选的兴趣类目（可选）
        冷启动召回 = 兴趣类目 + 热门兜底
        """
        results = []

        # 1. 兴趣类目召回
        if interest_cats:
            per_cat = max(1, k // (2 * len(interest_cats)))
            for cat in interest_cats:
                items = self.cat_items.get(cat, [])[:per_cat]
                for nid in items:
                    results.append({
                        "note_id": int(nid),
                        "score": 0.8,
                        "source": "cold_interest",
                    })

        # 2. 热门兜底
        hot_items = self.hot.recall(k=k)
        results += hot_items

        # 3. 去重
        by_id = {}
        for r in results:
            nid = r["note_id"]
            if nid not in by_id or r["score"] > by_id[nid]["score"]:
                by_id[nid] = r

        return sorted(by_id.values(), key=lambda x: -x["score"])[:k]