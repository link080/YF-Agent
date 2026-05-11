# YF-Agent - 智能闲鱼客服机器人系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![LLM Powered](https://img.shields.io/badge/LLM-powered-FF6F61)](https://platform.openai.com/)

专为闲鱼平台打造的AI值守解决方案，实现闲鱼平台7×24小时自动化值守，支持多专家协同决策、智能议价和上下文感知对话。 


## 🌟 核心特性

### 智能对话引擎
| 功能模块   | 技术实现            | 关键特性                                                     |
| ---------- | ------------------- | ------------------------------------------------------------ |
| 上下文感知 | 会话历史存储        | 轻量级对话记忆管理，完整对话历史作为LLM上下文输入            |
| 专家路由   | LLM prompt+规则路由 | 基于提示工程的意图识别 → 专家Agent动态分发，支持议价/技术/客服多场景切换 |

### 业务功能矩阵
| 模块     | 已实现                        | 规划中                       |
| -------- | ----------------------------- | ---------------------------- |
| 核心引擎 | ✅ LLM自动回复<br>✅ 上下文管理 | 🔄 情感分析增强               |
| 议价系统 | ✅ 阶梯降价策略                | 🔄 市场比价功能               |
| 技术支持 | ✅ 网络搜索整合                | 🔄 RAG知识库增强              |
| 工具系统 | ✅ 酒店查价Tool<br>✅ 批量爬取Tool | 🔄 RAG知识库增强 |
| 运维监控 | ✅ 基础日志                    | 🔄 钉钉集成<br>🔄  Web管理界面 |

## 🛠 工具系统

### 酒店数据工具（`utils/hotel_tools.py`）

专为酒店订房客服场景设计，提供三个 Tool：

| 工具名 | 触发场景 | 功能说明 |
| ------ | -------- | -------- |
| `search_hotel_price` | 查询酒店价格/预订 | 根据酒店名匹配ID，爬取华住会详情页获取所有房型及实时报价，结果自动保存至本地 Excel |
| `crawl_hotels` | 定时更新数据 | 批量爬取华住会指定城市酒店，刷新本地 Excel 缓存 |

### Tool 分发器（`tools.py`）

统一的 Tool 注册与执行接口，提供：

- `get_tool_schema()` — 返回 OpenAI Function Calling 格式的工具定义
- `has_tool(name)` — 检查工具是否存在
- `execute_tool(name, params)` — 分发器，路由到对应 handler 并截断大结果

## 🎨效果图
<div align="center">
  <img src="./images/demo1.png" width="600" alt="客服">
  <br>
  <em>图1: 客服随叫随到</em>
</div>


<div align="center">
  <img src="./images/demo2.png" width="600" alt="议价专家">
  <br>
  <em>图2: 阶梯式议价</em>
</div>

<div align="center">
  <img src="./images/demo3.png" width="600" alt="技术专家"> 
  <br>
  <em>图3: 技术专家上场</em>
</div>

<div align="center">
  <img src="./images/log.png" width="600" alt="后台log"> 
  <br>
  <em>图4: 后台log</em>
</div>

### 环境要求
- Python 3.8+

### 安装步骤
```bash
1. 克隆仓库
git clone https://github.com/link080/YF-Agent.git
cd YF-Agent

2. 安装依赖
```bash
pip install -r requirements.txt

# 如需使用酒店爬取工具，安装 Playwright 浏览器
playwright install chromium
```

3. 配置环境变量
创建一个 `.env` 文件，包含以下内容，也可直接重命名 `.env.example` ：
#必配配置
API_KEY=apikey通过模型平台获取
COOKIES_STR=填写网页端获取的cookie
MODEL_BASE_URL=模型地址
MODEL_NAME=模型名称
#可选配置
TOGGLE_KEYWORDS=接管模式切换关键词，默认为句号（输入句号切换为人工接管，再次输入则切换AI接管）
SIMULATE_HUMAN_TYPING=True/False #模拟人工回复延迟

注意：默认使用的模型是通义千问，如需使用其他API，请自行修改.env文件中的模型地址和模型名称；
COOKIES_STR自行在闲鱼网页端获取cookies(网页端F12打开控制台，选择Network，点击Fetch/XHR,点击一个请求，查看cookies)

4. 创建提示词文件prompts/*_prompt.txt（也可以直接将模板名称中的_example去掉），否则默认读取四个提示词模板中的内容
```

### 使用方法

运行主程序：
```bash
python main.py
```

### 自定义提示词

可以通过编辑 `prompts` 目录下的文件来自定义各个专家的提示词：

- `classify_prompt.txt`: 意图分类提示词
- `price_prompt.txt`: 价格专家提示词
- `tech_prompt.txt`: 技术专家提示词
- `default_prompt.txt`: 默认回复提示词

## 📝 更新日志

### 2026-04-30

**新增**
- 工具系统：支持 OpenAI Function Calling 格式的两个酒店数据工具（`search_hotel_price`、`crawl_hotels`）
- Tool 分发器 `tools.py`：统一的工具注册、校验与执行接口
- `BookingAgent` 预订 Agent：支持工具调用，可查询酒店信息和实时价格
- `BookingAgent` 专属提示词 `prompts/booking_prompt_example.txt`
- 意图路由新增 `booking` 类别，支持预订类关键词和日期正则匹配
- 分类提示词更新，新增 `booking` 和 `no_reply` 意图分类
- 终端对话测试脚本 `test_chat.py`

**优化**
- `PriceAgent` 接入工具系统，支持实时查价
- `search_hotel_price` 爬取逻辑重构：严格参考独立爬虫项目 `hotel_crawler.py`，增加 DOM 等待、多选择器回退、请求头补全
- `search_hotel_price` 合并 `search_hotels` + `get_hotel_price`：基于 `data/hotels.xlsx` 酒店ID数据库匹配，一步完成酒店查找与详情页爬取
- 价格提示词更新为单次 tool 调用指令："调用 search_hotel_price 工具获取华住会实时房价"

**修复**
- `context_manager.py` 新增 `_clean_content()` 方法，修复工具返回内容中 Unicode surrogates 字符导致的数据库写入错误
- `main.py` WebSocket 兼容：`extra_headers` → `additional_headers`（适配 websockets 13.x）
