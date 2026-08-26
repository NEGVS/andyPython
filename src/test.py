from dotenv import load_dotenv
import os
import google.genai

# 加载环境变量
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print(f"key={GOOGLE_API_KEY}")

if not GOOGLE_API_KEY:
    raise Exception("请在.env配置GOOGLE_API_KEY")

# 创建客户端
client = google.genai.Client(api_key=GOOGLE_API_KEY)
# 测试
resp = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="简单介绍RAG"
)
print("\nGemini输出：")
print(resp.text)