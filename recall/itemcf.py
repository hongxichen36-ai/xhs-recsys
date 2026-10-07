"""
ItemCF 召回（scipy 稀疏矩阵加速版）
用 scipy.sparse 把共现计算从 Python 三重循环变成一次矩阵乘法。
10000 用户 / 50000 item 数据下，构建时间 < 30 秒。
"""
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

ACTION_W = {"click": 1.0, "like": 2.0, "collect": 3.0, "comment": 4.0}
MAX_HISTORY = 20        # ← 从 50 降到 20，减少共现规模


class ItemCF:
    def __init__(self, actions_df, top_k_sim=30):
        self.top_k_sim = top_k_sim
        self.item_sim = {}
        self.user_history = {}
        self._build(actions_df)

    def _build(self, actions_df):
        import time
        t0 = time.time()

        # 1. 构造 user-item 稀疏矩阵
        print("[ItemCF] 1/3 构造 user-item 矩阵...")
        user_ids = actions_df["user_id"].values
        note_ids = actions_df["note_id"].values
        weights = np.array([ACTION_W[a] for a in actions_df["action"].values],
                           dtype=np.float32)

        # 重映射 user_id / note_id 到 0..N-1
        uniq_users = np.unique(user_ids)
        uniq_items = np.unique(note_ids)
        u2i = {u: i for i, u in enumerate(uniq_users)}
        n2i = {n: i for i, n in enumerate(uniq_items)}

        rows = np.array([u2i[u] for u in user_ids], dtype=np.int32)
        cols = np.array([n2i[n] for n in note_ids], dtype=np.int32)

        # 累加权重（同一 (u, i) 多次交互累加）
        X = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(uniq_users), len(uniq_items)),
        )
        # 合并重复项
        X.sum_duplicates()

        # 每个用户只保留权重最高的 MAX_HISTORY 个 item
        print(f"[ItemCF] 截断用户历史到 top-{MAX_HISTORY}...")
        X = self._truncate_topk(X, MAX_HISTORY)
        X.eliminate_zeros()

        # 保存 user_history
        X_csr = X.tocsr()
        for i, u in enumerate(uniq_users):
            row = X_csr.getrow(i)
            self.user_history[int(u)] = dict(
                zip(row.indices.tolist(), row.data.tolist())
            )

        print(f"[ItemCF] 1/3 完成，用户 {X.shape[0]}, item {X.shape[1]}, "
              f"耗时 {time.time()-t0:.1f}s")

        # 2. 共现矩阵 C = X.T @ X
        print("[ItemCF] 2/3 计算共现矩阵（矩阵乘法）...")
        C = (X.T @ X).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()

        # IIF 权重：每个 item 被多少用户交互过
        item_pop = np.asarray(X.sum(axis=0)).flatten() + 1
        iif = 1.0 / np.log(1 + item_pop)

        print(f"[ItemCF] 2/3 完成，共现非零元素 {C.nnz}, "
              f"耗时 {time.time()-t0:.1f}s")

        # 3. 逐行取 top-k 并归一化
        print("[ItemCF] 3/3 归一化 top-k...")
        for i in range(C.shape[0]):
            row = C.getrow(i)
            if row.nnz == 0:
                continue
            idx = row.indices
            vals = row.data * iif[idx]
            if len(vals) > self.top_k_sim:
                top_idx = np.argpartition(-vals, self.top_k_sim)[:self.top_k_sim]
                idx = idx[top_idx]
                vals = vals[top_idx]
            order = np.argsort(-vals)
            idx = idx[order]
            vals = vals[order]
            total = vals.sum() or 1.0
            note_ids_arr = uniq_items[idx]
            self.item_sim[i] = [
                (int(note_ids_arr[k]), float(vals[k] / total))
                for k in range(len(idx))
            ]

        print(f"[ItemCF] 全部完成，{len(self.item_sim)} items, "
              f"总耗时 {time.time()-t0:.1f}s")

    @staticmethod
    def _truncate_topk(X, k):
        """每行只保留前 k 个最大元素"""
        X = X.tocsr()
        indptr = X.indptr
        indices = X.indices
        data = X.data
        new_indices = []
        new_data = []
        new_indptr = [0]
        for i in range(X.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            row_data = data[start:end]
            row_idx = indices[start:end]
            if len(row_data) > k:
                top = np.argpartition(-row_data, k)[:k]
                row_data = row_data[top]
                row_idx = row_idx[top]
                order = np.argsort(-row_data)
                row_data = row_data[order]
                row_idx = row_idx[order]
            new_indices.extend(row_idx.tolist())
            new_data.extend(row_data.tolist())
            new_indptr.append(len(new_data))
        return csr_matrix(
            (np.array(new_data, dtype=np.float32),
             np.array(new_indices, dtype=np.int32),
             np.array(new_indptr, dtype=np.int32)),
            shape=X.shape,
        )

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