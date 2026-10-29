"""
千问 流式输出 FastAPI 服务（供 Spring Boot 调用，SSE 协议）
启动：uv run uvicorn src.qianWenProStream2:app --host 0.0.0.0 --port 8000
     或直接运行本文件：uv run python src/qianWenProStream2.py
接口：POST /api/v1/chat/stream   Content-Type: application/json   Accept: text/event-stream
请求：{"messages": [{"role": "user", "content": "你好"}]}
     多轮对话由调用方携带历史 messages，服务端无状态，可直接多实例水平扩容
响应：每个 SSE 事件的 data 为一行 JSON
     {"type": "reasoning", "content": "思考过程增量"}   ENABLE_THINKING=True 时先输出，调用方可展示或忽略
     {"type": "delta", "content": "回答增量"}
     {"type": "done", "usage": {"promptTokens": 10, "completionTokens": 20, "totalTokens": 30}}
     {"type": "error", "code": "RateLimitError", "message": "模型调用失败"}
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

# 国内：https://dashscope.aliyuncs.com/compatible-mode/v1
# 国际/新加坡：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"
HOST = "0.0.0.0"
PORT = 8000
# 流式场景下 timeout 是两个 chunk 之间的读超时，长回答不会被误杀；重试只发生在首包之前
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
# qwen3.7-plus 默认开启思考：首字约 8s、token 约 14 倍；关闭后首字约 1.4s.False
ENABLE_THINKING = False
# 单实例最大并发流数，超出直接 429 由调用方重试/降级，避免请求堆积拖垮实例和上游限流
MAX_CONCURRENT_STREAMS = 100
MAX_MESSAGES = 50
MAX_CONTENT_LENGTH = 20000
CHAT_STREAM_PATH = "/api/v1/chat/stream"
# X-Accel-Buffering: no 关闭 Nginx 缓冲，否则 SSE 会被攒成一整块才下发
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
ERROR_MESSAGE = "模型调用失败"
BUSY_MESSAGE = "服务繁忙，请稍后重试"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("qianWenProStream2")

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY 未配置，请检查 .env")

client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=MAX_CONTENT_LENGTH)

    @field_validator("content")
    @classmethod
    def check_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content 不能为空")
        return v


class ChatStreamRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def check_last_is_user(self) -> "ChatStreamRequest":
        if self.messages[-1].role != "user":
            raise ValueError("最后一条消息必须是 user")
        return self


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(serialization_alias="promptTokens")
    completion_tokens: int = Field(serialization_alias="completionTokens")
    total_tokens: int = Field(serialization_alias="totalTokens")


class SseEvent(BaseModel):
    type: Literal["reasoning", "delta", "done", "error"]
    content: str | None = None
    usage: TokenUsage | None = None
    code: str | None = None
    message: str | None = None


def to_sse(event: SseEvent) -> str:
    # JSON 会把换行转义为 \n，保证单个 data 行不会被 SSE 的空行分隔符截断
    return f"data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n"


async def chat_event_stream(messages: list[dict], request_id: str) -> AsyncIterator[str]:
    """
    信号量、上游连接都在生成器内获取并用 async with 释放：
    调用方断开时 Starlette 取消任务，CancelledError 不会被 except Exception 吞掉，上游连接随之关闭，不泄漏
    """
    async with semaphore:
        try:
            usage = None
            stream = await client.chat.completions.create(model=MODEL, messages=messages, stream=True, stream_options={"include_usage": True}, extra_body={"enable_thinking": ENABLE_THINKING})
            async with stream:
                async for chunk in stream:
                    # include_usage 的最后一个 chunk choices 为空，只带 usage
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        # 思考内容是 DashScope 扩展字段，SDK 未声明，用 getattr 兜底
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            yield to_sse(SseEvent(type="reasoning", content=reasoning))
                        if delta.content:
                            yield to_sse(SseEvent(type="delta", content=delta.content))
                    if chunk.usage:
                        usage = TokenUsage(prompt_tokens=chunk.usage.prompt_tokens or 0, completion_tokens=chunk.usage.completion_tokens or 0, total_tokens=chunk.usage.total_tokens or 0)
            yield to_sse(SseEvent(type="done", usage=usage))
            log.info("chat stream done, requestId=%s, messages=%d, usage=%s", request_id, len(messages), usage)
        except asyncio.CancelledError:
            log.info("chat stream client disconnected, requestId=%s", request_id)
            raise
        except Exception as e:
            log.warning("chat stream error, requestId=%s, messages=%d", request_id, len(messages), exc_info=True)
            yield to_sse(SseEvent(type="error", code=type(e).__name__, message=ERROR_MESSAGE))


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await client.close()


app = FastAPI(title="千问流式对话服务", lifespan=lifespan)


@app.post(CHAT_STREAM_PATH)
async def chat_stream(request: ChatStreamRequest, x_request_id: str | None = Header(default=None)) -> StreamingResponse:
    # 快速失败：满载时直接 429，不进入排队
    if semaphore.locked():
        raise HTTPException(status_code=429, detail=BUSY_MESSAGE)
    messages = [m.model_dump() for m in request.messages]
    return StreamingResponse(chat_event_stream(messages, x_request_id or "-"), media_type="text/event-stream", headers=SSE_HEADERS)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
