# 酒店对话信息抽取系统

RAG 检索增强酒店名 + 轻量 Qwen 大模型结构化抽取 + 日期规则引擎兜底。

## 架构

```
用户 query
  │
  ├─ 1. RAG 酒店名检索 ──── bge-m3 embedding + FAISS 向量库
  │                         模糊匹配、简称、口语、错别字容忍
  │                         从标准酒店列表中找到最匹配酒店
  │
  ├─ 2. Qwen 小模型抽取 ─── Qwen2.5-1.5B-Instruct (本地)
  │                         输入: query + 候选酒店
  │                         输出: 严格 JSON {hotel_name, checkin_date, checkout_date, confidence}
  │
  └─ 3. 日期规则引擎 ────── 纯 Python 规则, <10ms
                            支持所有数字格式、中文、相对日期(今天/明天/后天/周末)
                            标准化为 YYYY-MM-DD
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建酒店向量库 (首次运行或酒店列表更新时)
python build_hotel_index.py

# 3. 测试运行
python main.py

# 4. 作为模块调用
from main import extract_booking_info
result = extract_booking_info("帮我订汉庭西安边家村地铁站酒店明天的房间")
print(result)
# {"hotel_name": "汉庭西安边家村地铁站酒店(升级中)", "checkin_date": "2026-05-01", "checkout_date": "2026-05-02", "confidence": 0.92}
```

## 项目结构

```
models/RAG Entity Extraction/
├── requirements.txt          # Python 依赖
├── README.md                 # 本文档
├── config.py                 # 全局配置
├── hotel_index.py            # FAISS 酒店向量库 + RAG 检索
├── date_parser.py            # 日期规则引擎 (纯 Python)
├── llm_extractor.py          # Qwen 本地模型加载与推理
├── main.py                   # 主流程封装 + 测试
└── build_hotel_index.py      # 酒店向量库构建脚本
```

## 模型下载

### bge-m3 (已有)
路径: `D:\homework\project\YF-agent\YF-Agent\models\models--BAAI--bge-m3`

### Qwen2.5-1.5B-Instruct
```bash
# 方式1: huggingface-cli (需科学网络)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir D:/homework/project/YF-agent/YF-Agent/models/models--Qwen2.5-1.5B-Instruct

# 方式2: 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir D:/homework/project/YF-agent/YF-Agent/models/models--Qwen2.5-1.5B-Instruct

# 方式3: modelscope
pip install modelscope
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
  --local_dir D:/homework/project/YF-agent/YF-Agent/models/models--Qwen2.5-1.5B-Instruct
```

## 数据集特征

| 字段 | 说明 | 示例 |
|------|------|------|
| hotel_name | 标准酒店名(长名含品牌+城市+地标) | 汉庭西安边家村地铁站酒店(升级中) |
| checkin_date_text | 原始日期文本 | 12-20, 2025.3.13, 9月19号, 明天 |
| checkout_date_text | 同上 | 同上 |
| standard_checkin | 标准化后 YYYY-MM-DD | 2025-12-20 |
| question_intent | 意图标签 | availability/booking/price/inquiry |

## 日期格式覆盖

| 格式 | 示例 |
|------|------|
| 数字短横线 | 12-20, 04-13 |
| 数字短横线完整 | 2025-10-28 |
| 点分隔 | 2025.3.13, 2025.04.05 |
| 斜杠分隔 | 2025/11/24 |
| 中文 | 2025年08月16日, 9月19号 |
| 相对日期 | 今天, 明天, 后天, 大后天 |
| 星期 | 这个周五, 下周五, 本周末, 下周一 |

## 注意事项

- 所有模型从本地加载, 不联网下载
- 模型路径统一在 `config.py` 中配置
- 酒店列表从数据集的 `hotel_name` 列去重提取
