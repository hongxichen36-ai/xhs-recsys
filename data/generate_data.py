"""
模拟小红书风格笔记推荐数据（修复版 v2）
1000 用户 / 5000 笔记 / 20万行为日志

关键改动：
1. 先固定每个用户的兴趣偏好（users.csv）
2. 笔记带隐语义向量（latent），用于计算用户-笔记匹配度
3. 行为按用户兴趣 × 笔记 latent 的 softmax 采样，温度 0.1
4. 输出 users.csv / notes.csv / actions.csv 三张表
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_USER = 1000#用户数量
N_ITEM = 5000#笔记数量
N_ACTION = 200000#交互行为总量

CATEGORIES = ["美妆", "穿搭", "美食", "旅行", "职场",
              "学习", "健身", "宠物", "家居", "数码"]#笔记大类
N_CAT = len(CATEGORIES)#笔记大类数量

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
}#每个笔记大类下的子类


# ---------------- 1. 用户 ----------------
def generate_users():
    """
    每个用户固定 1~3 个偏好类目，并生成一个 10 维兴趣分布向量。
    偏好类目权重加随机强弱，避免所有偏好完全一样。
    """
    users = []
    for uid in range(N_USER):
        n_pref = np.random.randint(1, 4)#随机决定这个用户有几个偏好类目，randint(1, 4) 返回 1、2 或 3（不含 4）
        pref_idx = np.random.choice(N_CAT, size=n_pref, replace=False)#从 N_CAT的0~9 这 10 个类目索引里，不重复地选 n_pref 个，作为偏好类目。
        
        # 非偏好类目低权重
        interest = np.ones(N_CAT) * 0.05#长度为N_CAT的全部为0.05的向量
        # 偏好类目权重 0.8~1.5 随机，体现兴趣强弱
        for pi in pref_idx:
            interest[pi] = np.random.uniform(0.8, 1.5)
        interest /= interest.sum()#对该用户的喜好矩阵进行归一化，方便后续当概率用
        #前提是这个矩阵包含了用户对所有笔记大类的喜好


        users.append({
            "user_id": uid,
            "age": np.random.randint(18, 45),
            "gender": np.random.choice(["M", "F"], p=[0.35, 0.65]),#在男女中按概率分布随机选择
            "pref_cats": ",".join([CATEGORIES[i] for i in pref_idx]),#偏好的大类
            "interest": interest.tolist(),   # 临时列，不写入 csv
        })
    return pd.DataFrame(users)#每行是一个用户


# ---------------- 2. 笔记 ----------------
def generate_notes():
    """
    每篇笔记有类目 + 标签，并生成一个 10 维 latent 向量。
    latent 与类目强相关，用于计算用户-笔记匹配度。
    """
    notes = []
    for i in range(N_ITEM):
        cat = np.random.choice(CATEGORIES)
        cat_idx = CATEGORIES.index(cat)
        tags = np.random.choice(TAGS_PER_CAT[cat], size=2, replace=False).tolist()
        #一个笔记对应cat大类，标签是cat大类里面的两个小类

        # 隐语义向量：类目 one-hot + 噪声，归一化
        latent = np.random.randn(N_CAT) * 0.15#生成一个长度为N_CAT的向量
        latent[cat_idx] += 1.0#对于该笔记对应的大类对应的索引位置增加
        latent /= np.linalg.norm(latent)#除以向量的模长，这是归一化

        notes.append({
            "note_id": i,
            "author_id": np.random.randint(0, 500),
            "category": cat,
            "tags": ",".join(tags),
            "title": f"笔记_{i}_{cat}",
            "publish_ts": np.random.randint(1_700_000_000, 1_720_000_000),
            "latent": latent.tolist(),      # 临时列
        })
    return pd.DataFrame(notes)


# ---------------- 3. 行为 ----------------
def generate_actions(notes_df, users_df):#输入笔记表和用户表，返回行为表。
    """
    按用户批量采样行为：
      1. 一次性算好所有用户对所有笔记的匹配概率
      2. 每个用户用自己的概率分布一次性采样 n_u 篇笔记
      3. 行为类型和时间戳向量化生成
    温度从 0.3 改成 0.1，命中率从 ~45% 提到 ~88%。
    """
    note_latent = np.stack(notes_df["latent"].values)        # (N_ITEM, N_CAT)
    #不用stack就是(N_ITEM,)形状
    note_ids = notes_df["note_id"].values                    # (N_ITEM,)

    interest_matrix = np.stack(users_df["interest"].values)  # (N_USER, N_CAT)

    # 1. 所有用户 -> 所有笔记的匹配分
    scores = interest_matrix @ note_latent.T                 # (N_USER, N_ITEM)
    #@ np.matmul(interest_matrix, note_latent.T) np.dot(interest_matrix, note_latent.T)在二维情况都是矩阵乘积
    scores = scores / 0.1                                    # ✅ 温度 0.1
    #每一行表示用户对各个物品的喜好分数
    scores -= scores.max(axis=1, keepdims=True)
    #`axis=1`：沿着列方向（横向）做 max** → 对**每一行求最大值**
    #这里是(N_USER, N_ITEM)-(N_USER, 1)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)                # 每行和为 1

    # 2. 每个用户的行为数：多项分布
    actions_per_user = np.random.multinomial(
        N_ACTION, [1.0 / N_USER] * N_USER
    )
    ## 总共 N_ACTION 次行为，每个用户的概率是1.0 / N_USER

    # 3. 按用户批量采样
    actions = []
    for uid in range(N_USER):
        n_u = int(actions_per_user[uid])
        if n_u == 0:
            continue

        nids = np.random.choice(note_ids, size=n_u, p=probs[uid])
        #依据p对note_ids采样size次
        action_types = np.random.choice(
            ["click", "like", "collect", "comment"],
            size=n_u,
            p=[0.7, 0.15, 0.1, 0.05]
        )
        #依据p对交互行为列表采样size次
        timestamps = np.random.randint(
            1_700_000_000, 1_720_000_000, size=n_u
        )
 #每次交互操作（操作类型是act）某个物品（id是nid）在第ts时间
        for nid, act, ts in zip(nids, action_types, timestamps):#zip把三个等长列表逐位置配对，
            #返回一个生成器
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
#按 user_id 这一列的值分组。然后统计每个id的交互操作次数
    # ✅ 向量化算命中率，不用 apply，20 万行秒出
    note_cat = notes_df.set_index("note_id")["category"].to_dict()
    user_pref = users_df.set_index("user_id")["pref_cats"].to_dict()
    #把 note_id → category、user_id → pref_cats 做成字典，方便映射。

    actions_df = actions_df.copy()
    actions_df["note_cat"] = actions_df["note_id"].map(note_cat)
    #增加一列，是笔记id到大类的映射
    actions_df["pref"] = actions_df["user_id"].map(user_pref)#增加一列，是用户id到喜好大类的映射

    # 把 pref_cats 拆成集合，用集合包含判断
    pref_sets = users_df.set_index("user_id")["pref_cats"].str.split(",").apply(set)
    #首先user_df的index变成user_id,然后取出pref_cats列，用，切割字符串，然后转化为列表，最后把列表变成集合
    #`pref_sets` 是一个**Series，index=user_id，value = 该用户的偏好类目集合**
    actions_df["pref_set"] = actions_df["user_id"].map(pref_sets)

    # 逐行判断改成列表推导 + any，比 apply 快 10 倍
    hit_mask = np.array([
        cat in pset
        for cat, pset in zip(actions_df["note_cat"], actions_df["pref_set"])
    ])
    #把两个列按位置配对，每次取出一对 (cat, pset)：True=1，False=0
    rate = hit_mask.mean()
    print(f"  行为命中偏好类目比例: {rate * 100:.1f}%")
    print(f"  (随机基线约 20%~30%，健康值应 > 60%)")

    na = actions_df.groupby("note_id").size()
    print(f"  笔记平均被交互次数: {na.mean():.1f}, "
          f"未被交互笔记数: {N_ITEM - len(na)}")
    print(f"  行为类型分布:\n{actions_df['action'].value_counts()}")
    #.value_counts()统计每个交互出现的次数


# ---------------- 5. 主流程 ----------------
if __name__ == "__main__":
    # ✅ 用脚本所在目录作为基准，和当前工作目录无关
    BASE_DIR = Path(__file__).resolve().parent.parent   # xhs-recsys/
    out = BASE_DIR / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)

    users = generate_users()
    notes = generate_notes()
    actions = generate_actions(notes, users)

    users_clean = users.drop(columns=["interest"])
    notes_clean = notes.drop(columns=["latent"])

    users_clean.to_csv(out / "users.csv", index=False)
    notes_clean.to_csv(out / "notes.csv", index=False)
    actions.to_csv(out / "actions.csv", index=False)

    print(f"✅ 生成 {len(notes)} 条笔记, {len(users)} 个用户, {len(actions)} 条行为")
    print(f"📁 保存路径: {out.resolve()}")     # ✅ 打印绝对路径，方便确认
    validate(notes_clean, users_clean, actions)