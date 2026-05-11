"""LangGraph 状态定义"""
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages


class XianyuState(TypedDict):
    """贯穿整个图的共享状态"""
    user_msg: str
    item_desc: str
    context: List[Dict]
    formatted_context: str
    intent: str
    bargain_count: int
    messages: Annotated[list, add_messages]
    tool_loop_count: int
    response: str
    temperature: float
    use_tools: bool
    enable_search: bool
