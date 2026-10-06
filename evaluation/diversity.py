"""
多样性指标
- 类目熵
- 作者重复率
- 列表内类目相似度
"""
import numpy as np
from collections import Counter


def category_entropy(note_ids, note_cat):
    """类目熵：越大越多样"""
    cats = [note_cat.get(n) for n in note_ids if n in note_cat]
    if not cats:
        return 0.0
    counter = Counter(cats)
    total = len(cats)
    probs = [c / total for c in counter.values()]
    return float(-sum(p * np.log(p + 1e-9) for p in probs))


def author_repetition_rate(note_ids, note_author):
    """同作者重复率：0=没有重复，越接近 1 越差"""
    if len(note_ids) < 2:
        return 0.0
    authors = [note_author.get(n) for n in note_ids if n in note_author]
    if len(authors) < 2:
        return 0.0
    counter = Counter(authors)
    duplicates = sum(c - 1 for c in counter.values())
    return duplicates / (len(authors) - 1)


def intra_list_similarity(note_ids, note_cat):
    """列表内类目相似度：同类目对数 / 总对数"""
    if len(note_ids) < 2:
        return 0.0
    cats = [note_cat.get(n) for n in note_ids]
    same = 0
    total = 0
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            total += 1
            if cats[i] == cats[j]:
                same += 1
    return same / total if total > 0 else 0.0