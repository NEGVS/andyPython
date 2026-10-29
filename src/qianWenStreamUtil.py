"""
千问流式公共方法：读取模型流 + 打字机逐字输出
模型每个 chunk 通常含多个字，直接打印是一段一段蹦出来的；这里拆成单字并控制节奏
"""
import queue
import threading
import time
from typing import Generator, Iterator

from openai import OpenAI

# 事件类型，同时用作控制台输出的段落标题
REASONING = "思考过程"
ANSWER = "回答"
USAGE = "usage"
# 单字输出间隔（秒）
CHAR_INTERVAL_SECONDS = 0.05
# 待输出 chunk 积压超过该值时按比例加速，避免显示远远落后于生成
CATCH_UP_BACKLOG_CHUNKS = 200

_END = object()


def read_chat_stream(client: OpenAI, model: str, messages: list[dict], enable_thinking: bool) -> Generator[tuple[str, object], None, None]:
    """读取模型流，产出 (REASONING, 文本) / (ANSWER, 文本) / (USAGE, CompletionUsage)"""
    # stream_options.include_usage：最后一个 chunk 带 token 用量（该 chunk 的 choices 为空）
    with client.chat.completions.create(model=model, messages=messages, stream=True, stream_options={"include_usage": True}, extra_body={"enable_thinking": enable_thinking}) as stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                # 思考内容是 DashScope 扩展字段，SDK 未声明，用 getattr 兜底
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield REASONING, reasoning
                if delta.content:
                    yield ANSWER, delta.content
            if chunk.usage:
                yield USAGE, chunk.usage


def char_interval(backlog: int) -> float:
    """积压未超阈值时匀速输出，超过后间隔按 阈值/积压 等比缩短"""
    return CHAR_INTERVAL_SECONDS if backlog <= CATCH_UP_BACKLOG_CHUNKS else CHAR_INTERVAL_SECONDS * CATCH_UP_BACKLOG_CHUNKS / backlog


def typewriter(pieces: Generator[tuple[str, object], None, None]) -> Iterator[tuple[str, object]]:
    """
    把 pieces 的文本逐字产出，非 str 内容（如 usage）原样产出
    ============ 读写解耦 ============
    pieces 在后台线程迭代（建连、读流、关流都在后台），前台只负责逐字吐出：sleep 不阻塞网络读取，积压越多吐字越快
    调用方中断/关闭本生成器时通知后台线程，后台在下一个 chunk 到达时退出并关闭上游流
    """
    buffer: queue.Queue = queue.Queue()
    stop = threading.Event()

    def read() -> None:
        try:
            for piece in pieces:
                if stop.is_set():
                    break
                buffer.put(piece)
            # 提前 break 时触发 pieces 内 with 的退出，关闭上游 HTTP 流；正常结束时为空操作
            pieces.close()
        except Exception as e:
            buffer.put(e)
        finally:
            # 必须放 finally：保证前台 buffer.get() 一定能等到结束标记，不会永久阻塞
            buffer.put(_END)

    threading.Thread(target=read, daemon=True).start()
    try:
        while (item := buffer.get()) is not _END:
            if isinstance(item, Exception):
                raise item
            kind, content = item
            if not isinstance(content, str):
                yield item
                continue
            for ch in content:
                yield kind, ch
                time.sleep(char_interval(buffer.qsize()))
    finally:
        stop.set()
