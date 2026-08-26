from dotenv import load_dotenv
import os
from http import HTTPStatus
from dashscope import MultiModalConversation
import dashscope

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY 未配置，请检查 .env")

# 国内一般用这个（根据你的账号区域调整）
# dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
# 国际/新加坡：
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

resp = MultiModalConversation.call(
    api_key=DASHSCOPE_API_KEY,
    model="qwen3.7-plus",
    messages=[
        {
            "role": "user",
            "content": [                     # 注意：必须是 list
                {"text": "简单介绍RAG是什么"}
            ]
        }
    ],
    result_format="message"
)

if resp.status_code == HTTPStatus.OK:
    # 多模态返回的 content 是 list
    content = resp.output.choices[0].message.content
    if isinstance(content, list):
        print(content[0].get("text", content))
    else:
        print(content)
else:
    print(f"调用失败：{resp.code}, {resp.message}")