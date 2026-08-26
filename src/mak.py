from typing import List

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import chromadb
from dotenv import load_dotenv
import os
from openai import OpenAI

# 1. 创建 ChromaDB 客户端
client = chromadb.Client()
print(f"1. ------创建 ChromaDB 客户端")
print(client)
# 2. 创建 Collection
collection = client.get_or_create_collection(name="candidate_resumes")
print(f"2. 创建 Collection")
print(collection)

# 3. 加载 Embedding 模型
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# 文档分片
def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, "r", encoding="utf-8") as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]


chunks = split_into_chunks("/Users/andy_mac/Documents/CodeSpace/BackEnd/andyPython/doc/doc.md")


# 文档向量化
def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()


for i, chunk in enumerate(chunks):
    print(f"{i}: {chunk}\n")
    # test_embedding = embed_chunk(chunk)
    # print(len(test_embedding))
    # print(test_embedding)

embeddings = [embed_chunk(chunk) for chunk in chunks]
print(len(embeddings))
print(embeddings[0])


def save_embeddings(chunk: List[str], embeddings: List[List[float]]) -> None:
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )


# 保存数据
save_embeddings(chunks, embeddings)


# 召回
def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    print('召回')

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


print("===============")

query = '哆啦A梦使用的3个秘密道具分别是什么？'
print(query)
retrieved_chunks = retrieve(query, 5)
print('===============召回===============')
for i, chunk in enumerate(retrieved_chunks):
    print(f"{i}: {chunk}")

print('===============重排===============')


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


reranked_chunks = rerank(query, retrieved_chunks, 3)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")

print("=======发给大模型===========")
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


def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
            用户问题: {query}
            相关片段:
            {"\n\n".join(chunks)}
            请基于上述内容作答，不要编造信息。"""
    print(f"{prompt}\n\n---\n{prompt}")
    # content = "简单介绍RAG是什么"
    resp = client.chat.completions.create(
        model="qwen3.7-plus",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content


answer= generate(query,reranked_chunks)
print("==========大模型回答========")
print(answer)


# print("问题：")
# print(content)
#
# print("回答：")
# print(resp)
# print(resp.choices[0].message.content)
