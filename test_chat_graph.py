"""
终端多轮对话测试脚本（LangGraph 持久化版本）
支持 interrupt/resume 模式的连续对话
"""
import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from loguru import logger

from xianyu_graph import xianyu_graph
from context_manager import ChatContextManager
from langgraph.types import Command
from langgraph.errors import GraphInterrupt


def setup_logger(level: str = "DEBUG"):
    """配置日志输出"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def load_config():
    """加载环境变量配置"""
    if os.path.exists(".env"):
        load_dotenv()
        logger.info("已加载 .env 配置")

    required = ["API_KEY", "MODEL_BASE_URL", "MODEL_NAME"]
    for key in required:
        val = os.getenv(key)
        if not val:
            logger.error(f"缺少必要配置: {key}")
            sys.exit(1)
    logger.info(f"模型: {os.getenv('MODEL_NAME')} @ {os.getenv('MODEL_BASE_URL')}")


def build_item_description() -> str:
    """构建测试用商品信息描述"""
    return "当前商品的信息如下：" + json.dumps({
        "title": "全季酒店上海虹桥火车站店",
        "desc": "华住会旗下全季品牌酒店，位于上海虹桥火车站附近，交通便利",
        "price_range": "¥300-500",
        "total_stock": 50,
        "sku_details": [
            {"spec": "大床房", "price": 350, "stock": 20},
            {"spec": "双床房", "price": 400, "stock": 15},
            {"spec": "豪华套房", "price": 500, "stock": 5},
        ]
    }, ensure_ascii=False)


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print("  闲鱼智能客服 Agent - 终端对话测试 (LangGraph 持久化)")
    print("=" * 60)
    print("输入消息直接对话 | /reset 重置 | /quit 退出")
    print("-" * 60)


async def run_chat():
    """异步对话主循环（interrupt/resume 模式）"""
    setup_logger("DEBUG")
    load_config()

    # 清空历史对话数据库
    db_path = "data/test_chat_history.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"已清空历史对话: {db_path}")

    context_manager = ChatContextManager(
        max_history=100,
        db_path=db_path
    )

    chat_id = "test_chat_001"
    user_id = "test_user_001"
    item_id = "test_item_001"
    item_desc = build_item_description()

    # 图线程配置
    thread_id = f"thread_{chat_id}"
    graph_config = {"configurable": {"thread_id": thread_id}}

    # 预启动图，建立 interrupt 状态（首次 ainvoke 会停在 wait_for_input）
    try:
        await xianyu_graph.ainvoke({}, config=graph_config)
    except GraphInterrupt:
        logger.info("图已初始化，interrupt 状态就绪")

    print_banner()
    logger.info(f"会话已创建: {chat_id}")
    print()

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_msg = await loop.run_in_executor(None, lambda: input("你: ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_msg:
            continue

        if user_msg.lower() in ("/quit", "/exit"):
            print("退出测试")
            break

        if user_msg == "/reset":
            context_manager = ChatContextManager(
                max_history=100,
                db_path="data/test_chat_history.db"
            )
            logger.info(f"对话已重置: {chat_id}")
            print("[系统] 对话已重置，上下文已清空\n")
            continue

        # 构建恢复载荷
        context = context_manager.get_context_by_chat(chat_id)
        formatted_context = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in context if m["role"] in ("user", "assistant")
        )

        resume_payload = {
            "user_msg": user_msg,
            "item_desc": item_desc,
            "context": context,
            "formatted_context": formatted_context,
        }

        # 通过 interrupt/resume 模式生成回复
        try:
            logger.info(f"发送消息: {user_msg}")
            result = await asyncio.wait_for(
                xianyu_graph.ainvoke(
                    Command(resume=resume_payload),
                    config=graph_config,
                ),
                timeout=120,
            )
            bot_reply = result.get("response", "-")
            detected_intent = result.get("intent", "常规咨询")
            # confidence = result.get("confidence", 0)
            # logger.info(f"意图: {detected_intent}" + (f" (置信度: {confidence:.2f})" if confidence else ""))
        except GraphInterrupt:
            # 图 interrupt 后，从 checkpoint 读取响应
            snapshot = xianyu_graph.get_state(graph_config)
            bot_reply = snapshot.values.get("response", "-")
            detected_intent = snapshot.values.get("intent", "常规咨询")
            logger.info(f"意图: {detected_intent} (interrupt)")
        except asyncio.TimeoutError:
            logger.error("回复超时（120秒）")
            bot_reply = "抱歉，处理超时，请稍后再试"
            detected_intent = "error"
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            import traceback
            traceback.print_exc()
            bot_reply = f"抱歉，系统出现错误: {e}"
            detected_intent = "error"

        if bot_reply == "-":
            logger.info("消息被识别为无需回复类型")
            print("[系统] 无需回复\n")
            continue

        # 保存消息到上下文
        context_manager.add_message_by_chat(chat_id, user_id, item_id, "user", user_msg)
        context_manager.add_message_by_chat(chat_id, "seller", item_id, "assistant", bot_reply)

        if detected_intent == "议价砍价":
            context_manager.increment_bargain_count_by_chat(chat_id)

        a = xianyu_graph.get_state(graph_config)
        b = a.values.get("booking_status", "-")
        c = a.values.get("key_info", {})
        logger.info(f"状态: {b}")
        logger.info(f"关键信息: {c}")
        print(f"\n客服: {bot_reply}\n")
        


def main():
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
