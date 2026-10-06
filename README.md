# 小红书风格笔记推荐系统

> 多路召回 + LightGBM 排序 + MMR 多样性重排的全链路推荐系统

一个端到端的推荐系统实践项目，覆盖数据生成、多路召回、排序、重排、离线评估和在线服务全流程。

## 效果

| 指标 | 基础召回 | +向量召回 | +LightGBM 排序 | +MMR 重排 |
|---|---|---|---|---|
| Recall@100 | 0.1566 | 0.1552 | - | - |
| Coverage | 0.3964 | **0.4602** | - | - |
| NDCG@10 | - | - | 0.0592 | **0.0593** |
| Hit@10 | - | - | 0.4200 | 0.4200 |
| 类目熵 | - | - | 0.1912 | **0.2997** |
| 同作者重复率 | - | - | 0.0122 | **0.0022** |
| 推荐延迟 P50 | - | - | 77.5 ms | 77.5 ms |
| 缓存命中延迟 | - | - | - | **0.5 ms** |

**关键成果**：

- 多路召回覆盖率从 39.6% 提升到 46.0%（+16%）
- LightGBM 排序 NDCG@10 相比 RRF 直排提升 8.0%
- MMR 重排在几乎不损失相关性（NDCG +0.2%）的前提下，类目熵提升 56.7%，同作者重复率下降 81.8%
- 在线服务缓存加速 **155 倍**（77.5 ms → 0.5 ms）

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
│ 多路召回                            │
│  ├─ ItemCF（协同过滤）              │
│  ├─ 热门召回（兜底）                │
│  └─ 向量召回（MiniLM + FAISS）      │
│  ↓ RRF 融合                         │
└─────────────────────────────────────┘
   ↓ 200 个候选
┌─────────────────────────────────────┐
│ LightGBM 排序                       │
│  12 维特征（用户/笔记/交叉/召回分） │
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
| 数据处理 | Pandas, NumPy |
| 召回 | ItemCF, FAISS, sentence-transformers (MiniLM) |
| 排序 | LightGBM |
| 重排 | MMR (Maximal Marginal Relevance) |
| 评估 | Recall@K, NDCG@K, Hit@K, 类目熵, 作者重复率 |
| 服务 | FastAPI + Uvicorn |
| 前端 | Streamlit |
| 缓存 | 内存 TTL Cache（可替换为 Redis） |

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
```

### 2. 生成模拟数据

```bash
python data/generate_data.py
```

生成 1000 用户 / 5000 笔记 / 20 万条行为日志，输出到 `data/raw/`。

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
```

### 4. 启动 API 服务

```bash
uvicorn service.main:app --port 8000
```

打开 http://localhost:8000/docs 查看 API 文档（Swagger UI）。

### 5. 启动可视化 Demo

```bash
streamlit run demo/app.py
```

打开 http://localhost:8501 查看推荐效果。

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
      "note_id": 4404,
      "title": "笔记_4404_学习",
      "category": "学习",
      "tags": "tag4,cat4",
      "score": 0.87,
      "source": "mmr"
    }
  ],
  "latency_ms": 77.5,
  "from_cache": false
}
```

## 项目结构

```
xhs-recsys/
├── data/
│   ├── generate_data.py      # 模拟数据生成
│   └── raw/                  # 生成的 csv（gitignore）
├── recall/                   # 多路召回
│   ├── itemcf.py             # ItemCF 协同过滤
│   ├── hot.py                # 热门召回
│   ├── two_tower.py          # 向量召回（MiniLM + FAISS）
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
│   └── run_day4.py
├── requirements.txt
└── README.md
```

## 模块说明

### 多路召回

- **ItemCF**：基于用户行为共现计算 item-item 相似度，按行为类型加权（收藏 > 评论 > 点赞 > 点击）
- **热门召回**：全局热门 + 类目热门，作为兜底
- **向量召回**：MiniLM 编码笔记标题+标签+类目（384 维），用户向量由历史行为加权平均，FAISS 内积检索
- **融合策略**：RRF (Reciprocal Rank Fusion)，对分数尺度不敏感，避免归一化带来的长尾塌陷

### 排序

- **特征**：用户侧（活跃度、类目偏好、作者互动）、笔记侧（热度、类目热度）、交叉（偏好匹配）、召回分（三路 + RRF + 多路命中数），共 12 维
- **模型**：LightGBM 二分类 CTR 模型，正负样本比约 1:13

### 重排

- **MMR**：lambda=0.7，融合类目（0.3）、作者（0.4）、embedding（0.3）三层相似度
- **冷启动**：新用户走「兴趣类目 + 热门兜底」组合召回

### 在线服务

- **FastAPI**：自动生成 Swagger 文档，Pydantic 校验请求体
- **TTL 缓存**：内存缓存，5 分钟过期，缓存命中后延迟 < 1 ms
- **Streamlit Demo**：可视化推荐结果，展示服务状态和延迟

## 数据说明

本项目使用**模拟生成**的数据，不涉及任何真实用户数据。数据生成保证：

- 每个用户有稳定偏好类目（1~3 个）
- 行为按用户兴趣 × 笔记隐向量的 softmax 采样
- 命中偏好类目比例 91.2%（随机基线约 20%）

## 环境

- Python 3.10+
- 依赖见 `requirements.txt`
- 无需 GPU，CPU 可完整运行

## License

MIT