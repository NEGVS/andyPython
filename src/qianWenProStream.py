"""
千问 流式输出（单轮）
运行：uv run python src/qianWenProStream.py
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

# 国内：https://dashscope.aliyuncs.com/compatible-mode/v1
# 国际/新加坡：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"
# 单次请求超时（秒）、失败重试次数
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
# qwen3.7-plus 默认开启思考：首字约 8s、token 约 14 倍；关闭后首字约 1.4s，True
ENABLE_THINKING = False

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY 未配置，请检查 .env")

client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

content = "简单介绍RAG是什么,控制在 200个字内"
print("问题：")
print(content)

try:
    # stream_options.include_usage：最后一个 chunk 带 token 用量（该 chunk 的 choices 为空）
    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": ENABLE_THINKING},
    ) as stream:
        # 当前输出段：None / 思考过程 / 回答，切换时打印一次标题
        section = None
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # 思考内容是 DashScope 扩展字段，SDK 未声明，用 getattr 兜底
                reasoning = getattr(delta, "reasoning_content", None) if delta else None
                if reasoning:
                    if section != "思考过程":
                        section = "思考过程"
                        print(f"{section}：")
                    print(reasoning, end="", flush=True)
                if delta and delta.content:
                    if section != "回答":
                        print("\n\n回答：" if section else "回答：")
                        section = "回答"
                    print(delta.content, end="", flush=True)
            if chunk.usage:
                print(f"\n\n[token] 输入={chunk.usage.prompt_tokens} 输出={chunk.usage.completion_tokens} 合计={chunk.usage.total_tokens}")
except Exception as e:
    print(f"\n调用失败：{type(e).__name__}: {e}")
