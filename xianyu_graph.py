"""LangGraph 图定义 — 持久化对话循环 + 关键信息记忆"""
import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from loguru import logger

from graph_state import XianyuState
from tools_langgraph import get_langgraph_tools

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from inference import predict_intent


# ── Prompt 缓存 ──────────────────────────────────────────────

_PROMPTS: Dict[str, str] = {}


def _load_prompt(name: str) -> str:
    """加载提示词，优先 {name}.txt，否则 {name}_example.txt"""
    if name in _PROMPTS:
        return _PROMPTS[name]
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    target = os.path.join(prompt_dir, f"{name}.txt")
    if os.path.exists(target):
        path = target
    else:
        path = os.path.join(prompt_dir, f"{name}_example.txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    _PROMPTS[name] = content
    return content


# ── 安全过滤 ─────────────────────────────────────────────────

_BLOCKED = ["微信", "QQ", "支付宝", "银行卡", "线下"]


def _safe_filter(text: str) -> str:
    if not text:
        return ""
    return "请通过平台沟通" if any(p in text for p in _BLOCKED) else text


# ── 辅助函数 ─────────────────────────────────────────────────

def _format_history(context: list) -> str:
    msgs = [m for m in context if m["role"] in ("user", "assistant")]
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def _extract_bargain_count(context: list) -> int:
    for msg in context:
        content = msg.get("content") or ""
        if msg.get("role") == "system" and "议价次数" in content:
            match = re.search(r"议价次数[:：]\s*(\d+)", content)
            if match:
                return int(match.group(1))
    return 0


def _extract_key_info(text: str, existing: Dict[str, str]) -> Dict[str, str]:
    """从 LLM 回复中提取关键信息 JSON，合并到已有信息（非空覆盖）"""
    if not text:
        return existing or {}
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return existing or {}
    try:
        parsed = json.loads(match.group(1))
        if not isinstance(parsed, dict):
            return existing or {}
        merged = dict(existing or {})
        for k, v in parsed.items():
            if v and isinstance(v, str) and v.strip():
                merged[k] = v.strip()
        return merged
    except (json.JSONDecodeError, ValueError):
        return existing or {}


def _strip_key_info_json(text: str) -> str:
    """剥离 LLM 回复中的 ```json...``` 块，返回干净文本"""
    if not text:
        return ""
    return re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()


_KEY_INFO_INSTRUCTION = """【关键信息提取】（系统指令，必须执行）
每次回复末尾必须附带以下JSON块，用 ```json 包裹。此JSON不计入回复字数限制。
如果某项信息未提及则留空字符串，不要编造。
```json
{"hotel_name":"","room_type":"","check_in":"","check_out":"","guest_count":""}
```
示例：用户说"全季虹桥大床房5月10入住"，回复正文后附带：
```json
{"hotel_name":"全季虹桥","room_type":"大床房","check_in":"05-10","check_out":"","guest_count":""}
```"""


def _build_system_msg(prompt: str, item_desc: str, formatted_context: str,
                      bargain_count: int = 0, extra: str = "",
                      key_info: Dict[str, str] = None) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    parts = [f"【当前日期】{today}"]

    # 注入已知关键信息
    if key_info and any(key_info.values()):
        info_lines = []
        name_map = {
            "hotel_name": "酒店", "room_type": "房型",
            "check_in": "入住", "check_out": "离店", "guest_count": "人数",
        }
        for key, label in name_map.items():
            if key_info.get(key):
                info_lines.append(f"  {label}：{key_info[key]}")
        if info_lines:
            parts.append("【已知客户信息】\n" + "\n".join(info_lines))

    parts.extend([f"【商品信息】{item_desc}",
                  f"【你与客户对话历史】{formatted_context}", prompt,
                  _KEY_INFO_INSTRUCTION])
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _build_llm(temperature: float, use_tools: bool, enable_search: bool):
    kwargs = {}
    if enable_search:
        kwargs["extra_body"] = {"enable_search": True}
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-max"),
        openai_api_base=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        openai_api_key=os.getenv("API_KEY"),
        temperature=temperature,
        max_tokens=500,
        model_kwargs={"top_p": 0.8, **kwargs},
    )
    if use_tools:
        model = model.bind_tools(get_langgraph_tools())
    return model


# ── LLM 兜底意图识别 ─────────────────────────────────────────

async def _llm_fallback_intent(user_msg: str, formatted_context: str,
                                booking_status: str, key_info: Dict[str, str]) -> str:
    """当 BGE-M3 置信度 < 0.5 时，用 LLM 做二次意图识别"""
    key_info_str = json.dumps(key_info, ensure_ascii=False) if key_info else "{}"
    prompt = f"""你是意图分类器。根据对话上下文和 booking_status 判断用户意图。

【对话历史】
{formatted_context}

【当前消息】{user_msg}
【booking_status】{booking_status}
【已知信息】{key_info_str}

分类规则：
- 如果 booking_status=awaiting_final 且用户确认（"好的"、"确认"、"订吧"、"确认下单"）→ 返回 "confirm_booking"
- 如果用户明确表达预订意愿（"帮我订"、"我要预订"、"下单"、"订吧"）→ 返回 "confirm_booking"
- 如果用户在描述酒店信息（房型、日期、人数等）→ 返回 "info_update"
- 如果用户在砍价 → 返回 "议价砍价"
- 如果用户在问价格 → 返回 "查询价格"
- 如果用户取消（"算了"、"不订了"）→ 返回 "no_reply"
- 其他 → 返回 "常规咨询"

只返回分类名称，不要解释。"""

    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-max"),
        openai_api_base=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        openai_api_key=os.getenv("API_KEY"),
        temperature=0.1,
        max_tokens=20,
    )
    try:
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        result = (resp.content or "").strip()
        logger.info(f"LLM 兜底意图识别: {result}")
        return result
    except Exception as e:
        logger.warning(f"LLM 兜底意图识别失败: {e}")
        return "常规咨询"


# ── 节点函数 ─────────────────────────────────────────────────

async def intent_detection_node(state: XianyuState) -> dict:
    cleaned = re.sub(r"[^\w一-龥]", "", state["user_msg"])
    result = await asyncio.to_thread(predict_intent, cleaned)
    intent = result["intent"]
    confidence = result.get("confidence", 0)
    booking_status = state.get("booking_status", "active")

    if confidence < 0.8:
        logger.info(f"意图识别置信度较低: {confidence:.2f}，调用 LLM 兜底")
        intent = await _llm_fallback_intent(
            state["user_msg"],
            state.get("formatted_context", ""),
            booking_status,
            state.get("key_info", {}),
        )
    logger.info(f"意图: {intent}" + (f" (置信度: {confidence:.2f})" if confidence else ""))
    bargain_count = _extract_bargain_count(state["context"])
    return {"intent": intent, "bargain_count": bargain_count}


async def prepare_price_node(state: XianyuState) -> dict:
    prompt = _load_prompt("price_prompt")
    bc = state["bargain_count"]
    temp = min(0.3 + bc * 0.15, 0.9)
    extra = f"\n▲当前议价轮次：{bc}" if bc > 0 else ""
    key_info = state.get("key_info", {})
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"], bc, extra,
                                    key_info=key_info)
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": temp, "use_tools": True, "enable_search": False,
    }


async def prepare_booking_node(state: XianyuState) -> dict:
    prompt = _load_prompt("booking_prompt")
    key_info = state.get("key_info", {})
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"],
                                    key_info=key_info)
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": 0.4, "use_tools": True, "enable_search": False,
    }


async def prepare_bargain_node(state: XianyuState) -> dict:
    prompt = _load_prompt("contect_prompt")
    key_info = state.get("key_info", {})
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"],
                                    key_info=key_info)
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": 0.4, "use_tools": False, "enable_search": True,
    }


async def prepare_consult_node(state: XianyuState) -> dict:
    prompt = _load_prompt("default_prompt")
    key_info = state.get("key_info", {})
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"],
                                    key_info=key_info)
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": 0.7, "use_tools": False, "enable_search": False,
    }


async def llm_call_node(state: XianyuState) -> dict:
    model = _build_llm(state["temperature"], state["use_tools"],
                       state["enable_search"])
    response = await model.ainvoke(state["messages"])
    tool_loop_count = state.get("tool_loop_count", 0) + 1

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.debug(f"LLM 请求工具调用 (第{tool_loop_count}轮)")
        return {"messages": [response], "tool_loop_count": tool_loop_count}

    content = response.content or ""

    # 提取关键信息 JSON
    key_info = _extract_key_info(content, state.get("key_info", {}))
    if key_info != state.get("key_info", {}):
        logger.info(f"关键信息更新: {key_info}")

    # 剥离 JSON 块，安全过滤
    clean_content = _strip_key_info_json(content)
    clean_content = _safe_filter(clean_content)

    return {
        "response": clean_content,
        "key_info": key_info,
        "messages": [response],
        "tool_loop_count": tool_loop_count,
    }


async def safety_filter_node(state: XianyuState) -> dict:
    return {"response": _safe_filter(state["response"])}


async def no_reply_node(state: XianyuState) -> dict:
    return {"response": "-"}


async def wait_for_input_node(state: XianyuState) -> dict:
    """等待用户输入，通过 interrupt 暂停图执行"""
    # 超时检测（30分钟无新消息）
    last_ts = state.get("last_msg_timestamp", 0)
    if last_ts and (time.time() - last_ts) > 1800:
        logger.info("对话超时（30分钟），自动结束")
        return {"booking_status": "completed", "response": "-"}

    resume_value = interrupt(None)

    updates = {
        "tool_loop_count": 0,
        "last_msg_timestamp": time.time(),
    }

    # 仅在首次进入时设置 booking_status，不覆盖 check_continue 的判断结果
    if not state.get("booking_status"):
        updates["booking_status"] = "active"

    if isinstance(resume_value, dict):
        updates["user_msg"] = resume_value.get("user_msg", "")
        if "item_desc" in resume_value:
            updates["item_desc"] = resume_value["item_desc"]
        if "context" in resume_value:
            updates["context"] = resume_value["context"]
        if "formatted_context" in resume_value:
            updates["formatted_context"] = resume_value["formatted_context"]
    else:
        updates["user_msg"] = str(resume_value)

    return updates


async def clear_messages_node(state: XianyuState) -> dict:
    """每轮循环开始时清空消息缓冲"""
    return {"messages": []}


async def check_continue_node(state: XianyuState) -> dict:
    """根据 booking_status 决定下一步（纯路由，不做额外判断）"""
    return {}


async def confirm_booking_node(state: XianyuState) -> dict:
    """确认预订：从 key_info 读取信息，查询价格，生成确认回复"""
    # 如果已经是 awaiting_final，说明用户二次确认，预订完成
    if state.get("booking_status") == "awaiting_final":
        return {
            "response": "好的，已为您安排预订，祝您入住愉快！",
            "booking_status": "completed",
        }

    key_info = state.get("key_info", {})
    hotel = key_info.get("hotel_name", "")
    room = key_info.get("room_type", "")
    check_in = key_info.get("check_in", "")
    check_out = key_info.get("check_out", "")

    # 构建确认信息
    parts = ["好的，帮您确认预订信息："]
    if hotel:
        parts.append(f"酒店：{hotel}")
    if room:
        parts.append(f"房型：{room}")
    if check_in:
        parts.append(f"入住：{check_in}")
    if check_out:
        parts.append(f"离店：{check_out}")

    # 如果有酒店名和日期，尝试查询价格
    if hotel and check_in:
        try:
            from tools_langgraph import search_hotel_price
            price_result = await asyncio.to_thread(
                search_hotel_price.invoke,
                {"hotel_name": hotel, "check_in": check_in,
                 "check_out": check_out or check_in}
            )
            if price_result and "未找到" not in price_result:
                parts.append(f"\n{price_result}")
        except Exception as e:
            logger.warning(f"查询价格失败: {e}")

    parts.append("\n确认无误的话请回复\"确认下单\"，我帮您安排。")
    reply = "\n".join(parts)

    return {
        "response": reply,
        "booking_status": "awaiting_final",
    }


# ── 路由函数 ─────────────────────────────────────────────────

def route_intent(state: XianyuState) -> str:
    intent = state["intent"]
    if intent == "confirm_booking":
        return "confirm_booking"
    if intent == "info_update":
        return "prepare_consult"
    if intent == "no_reply":
        return "no_reply"
    if intent == "查询价格":
        return "prepare_price"
    if intent == "下单预订":
        return "prepare_booking"
    if intent == "议价砍价":
        return "prepare_bargain"
    return "prepare_consult"


def should_call_tools(state: XianyuState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        if state.get("tool_loop_count", 0) >= 3:
            logger.warning("工具调用已达 3 轮上限，强制结束")
            return "end"
        return "tools"
    return "end"


def should_continue(state: XianyuState) -> str:
    """判断图应该循环等待还是结束"""
    status = state.get("booking_status", "active")
    if status == "completed":
        return "end"
    if status == "confirming":
        return "confirm"
    return "wait"


# ── 构建图 ───────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    tool_node = ToolNode(get_langgraph_tools())

    g = StateGraph(XianyuState)

    # 新增节点
    g.add_node("wait_for_input", wait_for_input_node)
    g.add_node("clear_messages", clear_messages_node)
    g.add_node("check_continue", check_continue_node)
    g.add_node("confirm_booking", confirm_booking_node)

    # 原有节点
    g.add_node("intent_detection", intent_detection_node)
    g.add_node("prepare_price", prepare_price_node)
    g.add_node("prepare_booking", prepare_booking_node)
    g.add_node("prepare_bargain", prepare_bargain_node)
    g.add_node("prepare_consult", prepare_consult_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("tools", tool_node)
    g.add_node("safety_filter", safety_filter_node)
    g.add_node("no_reply", no_reply_node)

    # 边
    g.add_edge(START, "wait_for_input")
    g.add_edge("wait_for_input", "clear_messages")
    g.add_edge("clear_messages", "intent_detection")

    g.add_conditional_edges("intent_detection", route_intent, {
        "confirm_booking": "confirm_booking",
        "no_reply": "no_reply",
        "prepare_price": "prepare_price",
        "prepare_booking": "prepare_booking",
        "prepare_bargain": "prepare_bargain",
        "prepare_consult": "prepare_consult",
    })

    g.add_edge("prepare_price", "llm_call")
    g.add_edge("prepare_booking", "llm_call")
    g.add_edge("prepare_bargain", "llm_call")
    g.add_edge("prepare_consult", "llm_call")

    g.add_conditional_edges("llm_call", should_call_tools, {
        "tools": "tools",
        "end": "safety_filter",
    })
    g.add_edge("tools", "llm_call")

    g.add_edge("safety_filter", "check_continue")
    g.add_edge("no_reply", "check_continue")

    g.add_conditional_edges("check_continue", should_continue, {
        "wait": "wait_for_input",
        "confirm": "confirm_booking",
        "end": END,
    })

    g.add_edge("confirm_booking", "wait_for_input")

    return g


checkpointer = MemorySaver()
xianyu_graph = _build_graph().compile(checkpointer=checkpointer)
