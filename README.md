# 小红书风格笔记推荐系统

> 四路召回 + LightGBM 排序 + MMR 多样性重排的全链路推荐系统

一个端到端的推荐系统实践项目，覆盖数据生成、多路召回、排序、重排、离线评估和在线服务全流程。

## 效果

| 指标 | 值 |
|---|---|
| 数据规模 | 10000 用户 / 50000 笔记 / 196 万行为 |
| 召回路数 | 4 路（ItemCF + Hot + Vector + Cat） |
| 推荐延迟 P50 | 57 ms |
| 缓存命中延迟 | **0.5 ms** |
| 缓存加速比 | **114x** |

**关键成果**：

- 四路召回覆盖协同（ItemCF）、内容（Vector）、兜底（Hot）、类目热门（Cat）四个维度
- LightGBM 排序引入负采样，正负比从 1:267 修正到 1:10
- MMR 重排在 NDCG 无损前提下，列表类目熵提升 56.7%，同作者重复率下降 81.8%
- TTL 缓存命中延迟 0.5 ms，加速 114 倍
- 支持冷启动用户（行为 < 5 条）走「兴趣类目 + 热门」组合

## 架构

```
用户请求
   ↓
┌─────────────────────────────────────┐
│ FastAPI (/recommend)                │
│   ↓ 先查缓存                         │
│ TTL Cache ──命中──→ 直接返回         │
│   ↓ 未命中                           │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ 四路召回                            │
│  ├─ ItemCF（协同过滤，IIF 归一化）  │
│  ├─ Hot（全局 + 类目热门，兜底）    │
│  ├─ Vector（MiniLM + FAISS）        │
│  └─ Cat（用户偏好类目热门）         │
│  ↓ RRF 融合                         │
└─────────────────────────────────────┘
   ↓ 200 个候选
┌─────────────────────────────────────┐
│ LightGBM 排序                       │
│  12 维特征（用户/笔记/交叉/召回分） │
│  负采样到正负比 1:10                │
└─────────────────────────────────────┘
   ↓ Top-50
┌─────────────────────────────────────┐
│ MMR 多样性重排                      │
│  类目/作者/embedding 三层相似度     │
└─────────────────────────────────────┘
   ↓ Top-10
返回结果（写入缓存）
```

## 技术栈

| 层 | 技术 |
|---|---|
| 数据处理 | Pandas, NumPy, SciPy（稀疏矩阵） |
| 召回 | ItemCF（IIF 归一化）, FAISS, sentence-transformers (MiniLM), 类目热门 |
| 融合 | RRF (Reciprocal Rank Fusion) |
| 排序 | LightGBM（负采样 + 12 维特征） |
| 重排 | MMR (Maximal Marginal Relevance) |
| 评估 | Recall@K, NDCG@K, Hit@K, 类目熵, 作者重复率 |
| 服务 | FastAPI + Uvicorn |
| 前端 | Streamlit |
| 缓存 | 内存 TTL Cache（接口兼容 Redis） |

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
```

### 2. 生成模拟数据

```bash
python data/generate_data.py
```

生成 10000 用户 / 50000 笔记 / 196 万条行为日志，含 10% 冷用户和 10% 冷笔记。

### 3. 跑离线评估

```bash
# Day 1: ItemCF + 热门召回
python scripts/run_day1.py

# Day 2: 加入向量召回，对比覆盖率
python scripts/run_day2.py

# Day 3: LightGBM 排序
python scripts/run_day3.py

# Day 4: MMR 多样性重排
python scripts/run_day4.py

# 冷启动评估（热用户 vs 冷用户）
python scripts/run_cold_start.py
```

### 4. 启动 API 服务

```bash
uvicorn service.main:app --port 8000
```

打开 http://localhost:8000/docs 查看 Swagger 文档。

### 5. 启动可视化 Demo

```bash
streamlit run demo/app.py
```

打开 http://localhost:8501 查看推荐效果。

### 6. 多用户推荐检查

```bash
python scripts/check_rec.py
```

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务健康检查 |
| POST | `/recommend` | 获取 Top-K 推荐 |

**请求示例**：

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "k": 10}'
```

**响应示例**：

```json
{
  "user_id": 0,
  "items": [
    {
      "note_id": 38120,
      "title": "笔记_38120_旅行",
      "category": "旅行",
      "tags": "攻略,citywalk",
      "score": 0.87,
      "source": "mmr"
    }
  ],
  "latency_ms": 57.4,
  "from_cache": false
}
```

## 项目结构

```
xhs-recsys/
├── data/
│   ├── generate_data.py      # 模拟数据生成（含冷启动场景）
│   └── raw/                  # 生成的 csv（gitignore）
├── recall/                   # 多路召回
│   ├── itemcf.py             # ItemCF（scipy 稀疏矩阵 + IIF）
│   ├── hot.py                # 热门召回
│   ├── two_tower.py          # 向量召回（MiniLM + FAISS）
│   ├── cat_recall.py         # 类目召回
│   └── recall_manager.py     # RRF 融合
├── rank/                     # 排序
│   ├── features.py           # 12 维特征工程
│   └── fine_rank.py          # LightGBM 排序模型
├── rerank/                   # 重排
│   ├── mmr.py                # MMR 多样性重排
│   └── cold_start.py         # 冷启动策略
├── evaluation/               # 评估
│   ├── offline.py            # 召回评估
│   ├── ranking_metrics.py    # NDCG / Hit
│   └── diversity.py          # 类目熵 / 作者重复率
├── service/                  # 在线服务
│   ├── main.py               # FastAPI 入口
│   ├── pipeline.py           # 推荐主流程
│   ├── schemas.py            # Pydantic 模型
│   └── cache.py              # TTL 缓存
├── demo/                     # 前端
│   └── app.py                # Streamlit Demo
├── scripts/                  # 实验脚本
│   ├── run_day1.py
│   ├── run_day2.py
│   ├── run_day3.py
│   ├── run_day4.py
│   ├── run_cold_start.py
│   └── check_rec.py
├── requirements.txt
└── README.md
```

## 模块说明

### 四路召回

- **ItemCF**：基于用户行为共现计算 item-item 相似度，用 scipy 稀疏矩阵加速（50000 笔记 < 30 秒），IIF 归一化惩罚热门物品
- **热门召回**：全局热门 + 类目热门，兜底冷启动
- **向量召回**：MiniLM 编码笔记标题+标签+类目（384 维），用户向量由历史行为加权平均，FAISS 内积检索
- **类目召回**：对用户偏好类目，取该类目热度 top-200，覆盖 ItemCF/Vector 漏掉的"热门新内容"
- **融合策略**：RRF (Reciprocal Rank Fusion)，对分数尺度不敏感

### 排序

- **特征（12 维）**：用户侧（活跃度、类目偏好、作者互动）、笔记侧（热度、类目热度）、交叉（偏好匹配）、召回分（四路 + RRF + 多路命中数）
- **模型**：LightGBM 二分类 CTR，负采样到正负比 1:10
- **采样**：2000 用户 × k=200 候选，共约 1.7 万训练样本

### 重排

- **MMR**：lambda=0.7，融合类目（0.3）、作者（0.4）、embedding（0.3）三层相似度
- **效果**：NDCG 无损，类目熵 +56.7%，同作者重复率 -81.8%

### 冷启动

- **冷用户**（行为 < 5 条）：走「兴趣类目热门 + 全局热门」组合
- **冷笔记**：数据生成时预留 10% 低曝光笔记，模拟冷启动场景

### 在线服务

- **FastAPI**：自动生成 Swagger 文档，Pydantic 校验请求体
- **TTL 缓存**：内存缓存，5 分钟过期，缓存命中后延迟 < 1 ms
- **Streamlit Demo**：可视化推荐结果，展示服务状态和延迟

## 数据说明

本项目使用**模拟生成**的数据，不涉及任何真实用户数据。数据生成保证：

- 每个用户有稳定偏好类目（1~3 个），行为按兴趣 × 笔记隐向量的 softmax 采样
- 命中偏好类目比例 90.9%
- 含 10% 冷用户（行为 ≤5 条）
- 含 10% 冷笔记（低曝光）
- 行为类型分布：click 70% / like 15% / collect 10% / comment 5%

## 环境

- Python 3.10+
- 依赖见 `requirements.txt`
- 无需 GPU，CPU 可完整运行

## License

MIT