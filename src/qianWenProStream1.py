"""
千问 流式输出（打字机逐字输出） + 命令行多轮对话
运行：IDEA 直接运行，或在项目根目录 uv run python -m src.qianWenProStream1
命令：exit / quit 退出，clear 清空上下文
"""
import os
from contextlib import closing

from dotenv import load_dotenv
from openai import OpenAI

from src.qianWenStreamUtil import ANSWER, REASONING, read_chat_stream, typewriter

# 国内：https://dashscope.aliyuncs.com/compatible-mode/v1
# 国际/新加坡：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
# qwen3.7-plus 默认开启思考：首字约 8s、token 约 14 倍；关闭后首字约 1.4s
ENABLE_THINKING = True
# 思考过程用灰色输出，与回答区分
GRAY, RESET = "\033[90m", "\033[0m"
SYSTEM_PROMPT = "你是一个乐于助人的助手，回答简洁准确。"
# 最多保留最近 N 轮（一问一答为一轮），防止上下文无限增长超出模型窗口、token 费用失控
MAX_HISTORY_ROUNDS = 20
EXIT_COMMANDS = {"exit", "quit"}
CLEAR_COMMAND = "clear"


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未配置，请检查 .env")
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)


def trim_history(history: list[dict]) -> list[dict]:
    """只保留最近 MAX_HISTORY_ROUNDS 轮，截断点落在 user 消息上，保证 user/assistant 成对"""
    max_len = MAX_HISTORY_ROUNDS * 2
    return history[-max_len:] if len(history) > max_len else history


def stream_answer(client: OpenAI, messages: list[dict]) -> str:
    """
    流式打印思考过程（灰色）和回答，只返回回答文本：思考内容不写入历史，避免下一轮 token 翻倍
    Ctrl+C 可中断当前回答，已输出部分照常返回
    """
    parts = []
    in_reasoning = False
    try:
        # closing：Ctrl+C 时立即关闭 typewriter，通知后台线程停止读流
        with closing(typewriter(read_chat_stream(client, MODEL, messages, ENABLE_THINKING))) as chars:
            for kind, ch in chars:
                if kind == REASONING:
                    in_reasoning = True
                    # 每个字单独包颜色码，中断时终端不会残留灰色
                    print(f"{GRAY}{ch}{RESET}", end="", flush=True)
                elif kind == ANSWER:
                    if in_reasoning:
                        in_reasoning = False
                        print("\n")
                    parts.append(ch)
                    print(ch, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[已中断当前回答]", end="")
    print()
    return "".join(parts)


def main() -> None:
    client = build_client()
    history: list[dict] = []
    print(f"已连接 {MODEL}，输入 exit 退出，clear 清空上下文\n")

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("再见")
            return
        if question.lower() == CLEAR_COMMAND:
            history.clear()
            print("[上下文已清空]\n")
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": question}]
        print("千问：", end="", flush=True)
        try:
            answer = stream_answer(client, messages)
        except Exception as e:
            # 调用失败不写入历史，避免出现只有提问没有回答的脏上下文
            print(f"\n[调用失败] {type(e).__name__}: {e}\n")
            continue

        if answer:
            history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            history = trim_history(history)
        print()


if __name__ == "__main__":
    main()
