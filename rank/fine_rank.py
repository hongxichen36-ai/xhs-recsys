"""
LightGBM CTR 排序
"""
import lightgbm as lgb
import numpy as np
import pandas as pd


class FineRanker:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = None

    def train(self, train_df):
        X = train_df[self.feature_cols]
        y = train_df["label"]

        pos = int(y.sum())
        neg = len(y) - pos
        print(f"[LightGBM] 训练样本 {len(y)}, 正 {pos}, 负 {neg}, "
              f"正负比 1:{neg / max(pos, 1):.1f}")

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        print(f"[LightGBM] 训练完成")

    def predict(self, df):
        return self.model.predict_proba(df[self.feature_cols])[:, 1]

    def feature_importance(self):
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)