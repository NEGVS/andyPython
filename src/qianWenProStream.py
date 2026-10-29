"""
千问 流式输出（单轮，打字机逐字输出）
运行：IDEA 直接运行，或在项目根目录 uv run python -m src.qianWenProStream
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.qianWenStreamUtil import USAGE, read_chat_stream, typewriter

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
    # 当前输出段（思考过程 / 回答），切换时打印一次标题
    section = None
    for kind, value in typewriter(read_chat_stream(client, MODEL, [{"role": "user", "content": content}], ENABLE_THINKING)):
        if kind == USAGE:
            print(f"\n\n[token] 输入={value.prompt_tokens} 输出={value.completion_tokens} 合计={value.total_tokens}")
            continue
        if kind != section:
            print(f"{kind}：" if section is None else f"\n\n{kind}：")
            section = kind
        print(value, end="", flush=True)
except Exception as e:
    print(f"\n调用失败：{type(e).__name__}: {e}")
