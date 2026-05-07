# 酒店客服用户意图识别系统

基于 **BAAI/bge-m3** 嵌入模型 + **HDBSCAN** 无监督聚类 + **小样本微调** 的酒店客服意图分类系统。

## 意图分类

| 序号 | 意图 | 说明 | 示例 |
|------|------|------|------|
| 1 | 查询价格 | 询问房价、套餐价格、附加费用等 | "一间房多少钱" |
| 2 | 下单预订 | 订房、改期、取消、续住等操作 | "帮我订明天晚上的房间" |
| 3 | 议价砍价 | 要求优惠、折扣、比价议价 | "能打八折吗" |
| 4 | 常规咨询 | 设施、交通、服务政策等一般咨询 | "有停车场吗" |

## 快速开始

### 1. 环境安装

```bash
# 进入项目目录
cd D:/homework/project/YF-agent/YF-Agent/models

# 安装依赖
pip install -r requirements.txt

# 验证 GPU 可用（可选）
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### 2. 运行流程

```bash
# Step 1: 数据预处理 + BGE-M3向量化 + HDBSCAN聚类 + 样本筛选
python data_preprocessing.py

# Step 2: 模型微调训练
python train_model.py

# Step 3: 推理测试（交互式演示）
python inference.py
```

### 3. 代码集成调用

```python
from inference import predict_intent, predict_batch

# 单条实时推理
result = predict_intent("我要订一间大床房")
print(result)
# {'intent': '下单预订', 'confidence': 0.85, 'all_probs': {...}}

# 批量离线预测
results = predict_batch([
    "一间房多少钱",
    "能打八折吗",
    "你们酒店有停车场吗",
])
for r in results:
    print(f"{r['intent']} ({r['confidence']:.2%})")
```

## 项目结构

```
models/
├── config.py               # 全局配置（路径、超参、意图定义）
├── data_preprocessing.py   # 数据预处理 + 聚类 + 样本筛选
├── train_model.py          # 模型微调训练
├── inference.py            # 推理服务（实时+批量）
├── requirements.txt        # Python依赖
├── dataset/                # 原始数据
│   └── DeepUtteranceAggregation-master/.../ECD_sample/
└── output/                 # 自动生成的输出目录
    ├── cleaned_data.json   # 清洗后的所有数据
    ├── embeddings.npz      # BGE-M3向量
    ├── cluster_labels.json # 聚类标签
    ├── sampled_labels.json # 小样本标注数据
    ├── intent_model/       # 最终模型文件
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   ├── tokenizer files...
    │   └── intent_config.json
    └── checkpoints/        # 训练检查点
```

## 模型部署优化建议

### CPU 环境优化

```python
# 1. 使用 ONNX 导出 (推理加速 2-3x)
import torch
from inference import IntentPredictor

predictor = IntentPredictor()
dummy_input = (
    torch.zeros(1, 128, dtype=torch.long),
    torch.zeros(1, 128, dtype=torch.long),
)
torch.onnx.export(
    predictor.model, dummy_input,
    "output/intent_model/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
    },
)

# 2. 使用 ONNX Runtime 推理
import onnxruntime as ort
session = ort.InferenceSession("output/intent_model/model.onnx")
```

### GPU 环境优化

```python
# 1. 使用 FP16 推理 (显存减半，速度提升 1.5-2x)
with torch.cuda.amp.autocast():
    result = predictor.predict("房价多少")

# 2. 批量推理时使用更大的 batch_size
results = predictor.predict_batch(texts, batch_size=128)
```

### 生产环境建议

| 场景 | 建议 |
|------|------|
| 高并发 API | 使用 FastAPI + uvicorn，GPU 部署 |
| 边缘设备 | CPU + ONNX + 量化 (INT8) |
| 流式客服 | 单条推理 + 结果缓存 (LRU) |
| 离线分析 | 批量推理 + 多进程 |

## 意图识别落地注意事项

### 数据层面

- **领域适配**: ECD 数据集为电商领域，需补充酒店领域数据。本项目通过合成数据弥补，实际部署建议收集真实客服对话
- **意图边界**: "查询价格"与"议价砍价"存在天然重叠，建议在业务层定义明确的跳转规则
- **未知意图**: 聚类可能发现新的意图模式，应定期 review 噪声点数据

### 模型层面

- **小样本限制**: 120-200 条标注数据适合冷启动，但长尾场景覆盖不足，建议持续积累标注数据
- **置信度阈值**: 生产环境应设置置信度下限 (如 0.6)，低于阈值转人工处理
- **持续学习**: 建议每月用新数据微调一次，更新模型

### 系统层面

- **监控告警**: 记录所有预测结果，统计意图分布变化，发现漂移及时告警
- **降级策略**: 模型不可用时降级为关键词匹配
- **A/B 测试**: 新模型上线前用历史数据进行离线对比测试
