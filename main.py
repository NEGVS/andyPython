import chromadb
from sentence_transformers import SentenceTransformer

# 1. 创建 ChromaDB 客户端
client = chromadb.Client()

print(f"1. ====创建 ChromaDB 客户端")
print(client)
# 2. 创建 Collection
collection = client.get_or_create_collection(name="candidate_resumes")
print(f"2. 创建 Collection")
print(collection)
# 3. 加载 Embedding 模型
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print(f"3. 加载 Embedding 模型")
print(model)

# 4. 准备数据
# 4. 准备数据
documents = [
    "5年Java开发经验，熟悉Spring Boot、Redis、MySQL、Kafka",
    "3年前端开发经验，熟悉Vue3、JavaScript、TypeScript",
    "4年AI开发经验，熟悉Python、RAG、LangChain、大模型",
    "6年Go后端开发经验，熟悉Gin、gRPC、PostgreSQL、RabbitMQ",
    "2年测试开发经验，熟悉Python、Pytest、JMeter、接口自动化",
    "7年运维开发经验，熟悉Docker、K8s、Shell、Prometheus",
    "4年大数据开发经验，熟悉Spark、Flink、Hive、Hadoop",
    "3年小程序开发经验，熟悉UniApp、微信小程序、TS、Node.js",
    "5年安全开发经验，熟悉Python、漏洞挖掘、渗透测试、Web安全",
    "4年产品经理经验，熟悉需求分析、PRD撰写、原型设计、用户调研",
    "6年PHP后端经验，熟悉Laravel、MySQL、Redis、Nginx",
    "3年算法开发经验，熟悉Python、PyTorch、CV、图像识别",
    "5年Node.js后端经验，熟悉Express、NestJS、MongoDB、Redis",
    "2年移动端开发经验，熟悉Flutter、Dart、Android、iOS跨端开发",
    "4年数据库DBA经验，熟悉MySQL调优、Redis、分库分表、备份恢复"
]

ids = [
    "candidate_001",
    "candidate_002",
    "candidate_003",
    "candidate_004",
    "candidate_005",
    "candidate_006",
    "candidate_007",
    "candidate_008",
    "candidate_009",
    "candidate_010",
    "candidate_011",
    "candidate_012",
    "candidate_013",
    "candidate_014",
    "candidate_015"
]

# 5. 生成向量
embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()
print(f"5. 生成向量")
print(embeddings)
for k in embeddings:
    print(k)
    print(len(k))
# 6. 写入 ChromaDB
re = collection.upsert(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)
print(f"6. 写入 ChromaDB")
print(re)
# 7. 用户查询
query = "寻找熟悉Java微服务和Redis的后端工程师"
print(f"7. 用户查询")
print(query)
# 8. 生成查询向量
query_embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()
print(f"8. 生成查询向量")
# 9. 相似度搜索
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

print(f"9. 相似度搜索")
# 10. 输出结果
for i, document in enumerate(results["documents"][0]):
    print(f"排名 {i + 1}")
    print(f"候选人：{results['ids'][0][i]}")
    print(f"简历：{document}")
    print(f"距离：{results['distances'][0][i]}")
    print("-" * 50)
