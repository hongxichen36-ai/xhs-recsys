"""
Day 1 一键脚本：加载数据 -> 构建召回 -> 离线评估
"""
import sys
from pathlib import Path

# 让脚本能 import 项目模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from recall.itemcf import ItemCF
from recall.hot import HotRecall
from recall.recall_manager import RecallManager
from evaluation.offline import split_train_test, evaluate_recall


def main():
    # 1. 加载数据
    data_dir = ROOT / "data" / "raw"
    actions = pd.read_csv(data_dir / "actions.csv")
    notes = pd.read_csv(data_dir / "notes.csv")
    print(f"加载: {len(actions)} 行为, {len(notes)} 笔记")

    # 2. 划分训练/测试
    train_df, test_df = split_train_test(actions, test_ratio=0.2)
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

    # 3. 构建召回
    itemcf = ItemCF(train_df, top_k_sim=50)
    hot = HotRecall(train_df, notes, top_k=500)
    manager = RecallManager(itemcf, hot)

    # 4. 离线评估
    evaluate_recall(manager, train_df, test_df, n_users=200, k=100)

    # 5. 看一个用户的推荐结果
    uid = 0
    recs = manager.recall(uid, k=10)
    print(f"\n👤 用户 {uid} 的 Top-10 推荐:")
    for r in recs:
        note_cat = notes.set_index("note_id").loc[r["note_id"], "category"]
        print(f"  note_id={r['note_id']:5d}  "
              f"score={r['score']:.4f}  "
              f"source={r['source']:8s}  "
              f"category={note_cat}")


if __name__ == "__main__":
    main()