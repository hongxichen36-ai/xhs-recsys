"""
内容向量召回（方案 A：不训练 MLP）
- 笔记：MiniLM 编码标题+标签+类目 -> 384 维，归一化
- 用户：历史笔记 embedding 按行为加权平均 -> 归一化
- 检索：FAISS IndexFlatIP
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict


ACTION_WEIGHT = {
    "click": 1.0,
    "like": 2.0,
    "collect": 3.0,
    "comment": 4.0,
}


class ContentVectorRecall:
    def __init__(self, notes_df,
                 model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.dim = 384
        self.encoder = SentenceTransformer(model_name)
        self.note_ids = notes_df["note_id"].values
        self.note_emb = None
        self.index = None
        self._build(notes_df)

    def _build(self, notes_df):
        texts = [
            f"{row['title']} {str(row['tags']).replace(',', ' ')} {row['category']}"
            for _, row in notes_df.iterrows()
        ]
        embs = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.note_emb = embs
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

        self.nid2row = {int(n): i for i, n in enumerate(self.note_ids)}
        print(f"[VectorRecall] indexed {len(self.note_ids)} notes, dim={self.dim}")

    def build_user_vector(self, history):
        """history: [{"note_id":..., "action":...}, ...]"""
        vecs, weights = [], []
        for h in history:
            row = self.nid2row.get(int(h["note_id"]))
            if row is None:
                continue
            vecs.append(self.note_emb[row])
            weights.append(ACTION_WEIGHT.get(h["action"], 1.0))

        if not vecs:
            return None

        vecs = np.stack(vecs)
        weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
        user_vec = (vecs * weights).sum(axis=0) / weights.sum()
        norm = np.linalg.norm(user_vec)
        if norm > 1e-8:
            user_vec = user_vec / norm
        return user_vec.astype(np.float32)

    def recall(self, history, k=100, exclude_items=None):
        user_vec = self.build_user_vector(history)
        if user_vec is None:
            return []

        scores, idx = self.index.search(user_vec.reshape(1, -1), k * 2)
        exclude = set(exclude_items or [])

        results = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            nid = int(self.note_ids[i])
            if nid in exclude:
                continue
            results.append({
                "note_id": nid,
                "score": float(s),
                "source": "vector",
            })
            if len(results) >= k:
                break
        return results