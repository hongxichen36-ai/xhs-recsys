"""
模拟小红书风格笔记推荐数据 v3
10000 用户 / 50000 笔记 / 200 万行为
含 10% 冷用户（行为 ≤5 条）+ 10% 冷笔记（被交互 ≤3 次）
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_USER = 10000
N_ITEM = 50000
N_ACTION = 2_000_000

COLD_USER_RATIO = 0.10     # 10% 用户是冷用户
COLD_NOTE_RATIO = 0.10     # 10% 笔记是冷笔记

CATEGORIES = ["美妆", "穿搭", "美食", "旅行", "职场",
              "学习", "健身", "宠物", "家居", "数码"]
N_CAT = len(CATEGORIES)

TAGS_PER_CAT = {
    "美妆": ["口红", "粉底", "眼影", "护肤", "彩妆"],
    "穿搭": ["通勤", "休闲", "连衣裙", "西装", "小个子"],
    "美食": ["火锅", "烘焙", "探店", "减脂餐", "早餐"],
    "旅行": ["攻略", "民宿", "海边", "雪山", "citywalk"],
    "职场": ["简历", "面试", "升职", "跳槽", "Python"],
    "学习": ["考研", "英语", "考证", "笔记法", "效率"],
    "健身": ["减脂", "增肌", "瑜伽", "跑步", "居家"],
    "宠物": ["猫", "狗", "养猫日常", "狗粮", "绝育"],
    "家居": ["装修", "收纳", "北欧", "租房改造", "绿植"],
    "数码": ["手机", "耳机", "笔记本", "键盘", "相机"],
}


# ---------------- 1. 用户 ----------------
def generate_users():
    users = []
    for uid in range(N_USER):
        is_cold = int(np.random.rand() < COLD_USER_RATIO)
        n_pref = np.random.randint(1, 4)
        pref_idx = np.random.choice(N_CAT, size=n_pref, replace=False)

        interest = np.ones(N_CAT) * 0.05
        for pi in pref_idx:
            interest[pi] = np.random.uniform(0.8, 1.5)
        interest /= interest.sum()

        users.append({
            "user_id": uid,
            "age": np.random.randint(18, 45),
            "gender": np.random.choice(["M", "F"], p=[0.35, 0.65]),
            "pref_cats": ",".join([CATEGORIES[i] for i in pref_idx]),
            "is_cold": is_cold,           # ← 冷用户标记
            "interest": interest.tolist(),
        })
    return pd.DataFrame(users)


# ---------------- 2. 笔记 ----------------
def generate_notes():
    notes = []
    # 冷笔记：最后 COLD_NOTE_RATIO 比例的笔记是冷的
    n_cold = int(N_ITEM * COLD_NOTE_RATIO)
    cold_start_id = N_ITEM - n_cold

    for i in range(N_ITEM):
        cat = np.random.choice(CATEGORIES)
        cat_idx = CATEGORIES.index(cat)
        tags = np.random.choice(TAGS_PER_CAT[cat], size=2, replace=False).tolist()

        latent = np.random.randn(N_CAT) * 0.15
        latent[cat_idx] += 1.0
        latent /= np.linalg.norm(latent)

        # 冷笔记曝光概率极低
        exposure_scale = 1.0
        if i >= cold_start_id:
            exposure_scale = 0.02       # 冷笔记曝光概率乘 0.02
        elif i >= int(cold_start_id * 0.85):
            exposure_scale = 0.3        # 次冷笔记

        notes.append({
            "note_id": i,
            "author_id": np.random.randint(0, 5000),
            "category": cat,
            "tags": ",".join(tags),
            "title": f"笔记_{i}_{cat}",
            "publish_ts": np.random.randint(1_700_000_000, 1_720_000_000),
            "is_cold": int(i >= cold_start_id),
            "exposure_scale": exposure_scale,
            "latent": latent.tolist(),
        })
    return pd.DataFrame(notes)


# ---------------- 3. 行为 ----------------
def generate_actions(notes_df, users_df):
    note_latent = np.stack(notes_df["latent"].values)
    note_ids = notes_df["note_id"].values
    exposure_scale = notes_df["exposure_scale"].values
    interest_matrix = np.stack(users_df["interest"].values)
    is_cold_user = users_df["is_cold"].values

    # 1. 匹配分 + 曝光缩放
    scores = interest_matrix @ note_latent.T                  # (N_USER, N_ITEM)
    scores = scores / 0.1
    scores += np.log(exposure_scale + 1e-9)
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)

    # 2. 分配行为数（按实际冷/热用户数，避免索引越界）
    actual_cold = int(is_cold_user.sum())
    actual_hot = N_USER - actual_cold
    print(f"  实际热用户: {actual_hot}, 冷用户: {actual_cold}")

    hot_total = int(N_ACTION * 0.98)

    hot_actions = np.random.multinomial(
        hot_total, [1.0 / actual_hot] * actual_hot
    ) if actual_hot > 0 else np.array([], dtype=int)

    cold_actions = np.random.randint(1, 6, size=actual_cold)

    actions_per_user = np.zeros(N_USER, dtype=int)
    hot_i, cold_i = 0, 0
    for uid in range(N_USER):
        if is_cold_user[uid]:
            actions_per_user[uid] = cold_actions[cold_i]
            cold_i += 1
        else:
            actions_per_user[uid] = hot_actions[hot_i]
            hot_i += 1

    # 3. 按用户批量采样
    actions = []
    for uid in range(N_USER):
        n_u = int(actions_per_user[uid])
        if n_u == 0:
            continue
        nids = np.random.choice(note_ids, size=n_u, p=probs[uid])
        action_types = np.random.choice(
            ["click", "like", "collect", "comment"],
            size=n_u,
            p=[0.7, 0.15, 0.1, 0.05]
        )
        timestamps = np.random.randint(
            1_700_000_000, 1_720_000_000, size=n_u
        )
        for nid, act, ts in zip(nids, action_types, timestamps):
            actions.append({
                "user_id": int(uid),
                "note_id": int(nid),
                "action": str(act),
                "timestamp": int(ts),
            })
    return pd.DataFrame(actions)

# ---------------- 4. 质量校验 ----------------
def validate(notes_df, users_df, actions_df):
    print("\n📊 数据质量校验:")

    ua = actions_df.groupby("user_id").size()
    print(f"  用户平均行为数: {ua.mean():.1f}, 最少 {ua.min()}, 最多 {ua.max()}")

    # 冷用户统计
    cold_uids = users_df[users_df["is_cold"] == 1]["user_id"].tolist()
    cold_ua = ua.reindex(cold_uids).fillna(0)
    print(f"  冷用户数: {len(cold_uids)}, 平均行为数: {cold_ua.mean():.1f}")

    # 冷笔记统计
    na = actions_df.groupby("note_id").size()
    cold_nids = notes_df[notes_df["is_cold"] == 1]["note_id"].tolist()
    cold_na = na.reindex(cold_nids).fillna(0)
    print(f"  冷笔记数: {len(cold_nids)}, 平均被交互: {cold_na.mean():.1f}, "
          f"0 交互数: {(cold_na == 0).sum()}")

    # 命中率
    note_cat = notes_df.set_index("note_id")["category"].to_dict()
    pref_sets = users_df.set_index("user_id")["pref_cats"].str.split(",").apply(set)
    actions_df = actions_df.copy()
    actions_df["note_cat"] = actions_df["note_id"].map(note_cat)
    actions_df["pref_set"] = actions_df["user_id"].map(pref_sets)
    hit_mask = np.array([
        cat in pset
        for cat, pset in zip(actions_df["note_cat"], actions_df["pref_set"])
    ])
    rate = hit_mask.mean()
    print(f"  行为命中偏好类目比例: {rate * 100:.1f}%")
    print(f"  笔记平均被交互次数: {na.mean():.1f}, "
          f"未被交互笔记数: {N_ITEM - len(na)}")
    print(f"  行为类型分布:\n{actions_df['action'].value_counts()}")


# ---------------- 5. 主流程 ----------------
if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "raw"
    out.mkdir(parents=True, exist_ok=True)

    print("生成用户...")
    users = generate_users()
    print("生成笔记...")
    notes = generate_notes()
    print("生成行为（200 万条，约需 30 秒）...")
    actions = generate_actions(notes, users)

    users_clean = users.drop(columns=["interest"])
    notes_clean = notes.drop(columns=["exposure_scale", "latent"])

    users_clean.to_csv(out / "users.csv", index=False)
    notes_clean.to_csv(out / "notes.csv", index=False)
    actions.to_csv(out / "actions.csv", index=False)

    print(f"✅ 生成 {len(notes)} 笔记, {len(users)} 用户, {len(actions)} 行为")
    print(f"📁 保存路径: {out.resolve()}")
    validate(notes_clean, users_clean, actions)