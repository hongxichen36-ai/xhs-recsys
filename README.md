# 小红书风格笔记推荐系统

> 多路召回 + 多目标排序 + 多样性重排的全链路推荐系统

## 技术栈
- 数据：Pandas + 模拟数据
- 召回：ItemCF + 热门召回 + FAISS 双塔
- 排序：LightGBM 多目标
- 重排：MMR 多样性 + 业务规则
- 服务：FastAPI + Redis
- 前端：Streamlit

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成模拟数据
python data/generate_data.py

# 3. 跑离线评估
python evaluation/offline.py
```

## 项目结构
```
xhs-recsys/
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 特征
│   └── generate_data.py      # 模拟数据生成
├── recall/
│   ├── itemcf.py             # ItemCF 召回
│   ├── hot.py                # 热门召回
│   └── recall_manager.py     # 多路召回统一入口
├── rank/                     # Day 3
├── rerank/                   # Day 4
├── evaluation/
│   └── offline.py            # 离线评估
├── service/                  # Day 5
├── demo/                     # Day 5
├── models/
└── requirements.txt
```
