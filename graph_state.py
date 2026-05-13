"""LangGraph 状态定义"""
from typing import TypedDict, List, Dict, Optional


class XianyuState(TypedDict):
    """贯穿整个图的共享状态"""
    user_msg: str
    item_desc: str
    context: List[Dict]
    formatted_context: str
    intent: str
    bargain_count: int
    messages: list
    tool_loop_count: int
    response: str
    temperature: float
    use_tools: bool
    enable_search: bool
    booking_status: str
    last_msg_timestamp: float
    key_info: Dict[str, str]
