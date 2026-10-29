我本机已经配置好了，以下代码可以正常运行的，你可以参考使用
```python
from dotenv import load_dotenv
import os
from openai import OpenAI

from src.qianWen import content

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY 未配置，请检查 .env")
print("api_key：")
print(api_key)
client = OpenAI(
    api_key=api_key,
    # 国内：
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 国际/新加坡：
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
print("client：")
print(client)

content = "简单介绍RAG是什么,控制在 100个字内"
resp = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[
        {"role": "user", "content": content}
    ]
)
print("问题：")
print(content)

print("回答：")
print(resp)
print(resp.choices[0].message.content)

```