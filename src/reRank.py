import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# 1. 配置
# ============================================================

BASE_DIR = Path(__file__).parent

RESUME_DIR = BASE_DIR / "data" / "resumes"

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "candidate_resumes"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RERANK_MODEL_NAME = (
    "BAAI/bge-reranker-base"
)

CHUNK_SIZE = 300

CHUNK_OVERLAP = 50

VECTOR_TOP_K = 10

RERANK_TOP_K = 10

CANDIDATE_TOP_K = 5


# ============================================================
# 2. 加载环境变量
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# 3. JD
# ============================================================

JD = """
招聘岗位：Java高级开发工程师

岗位要求：

1. 5年以上Java开发经验
2. 熟悉Spring Boot
3. 熟悉Spring Cloud微服务架构
4. 熟悉MySQL数据库
5. 熟悉Redis缓存
6. 熟悉Kafka消息队列
7. 有Elasticsearch开发经验优先
8. 有大型企业系统开发经验
9. 熟悉Docker
"""


# ============================================================
# 4. 文本切块
# ============================================================

def chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
):
    """
    简单文本切块

    参数：
    text       原始文本
    chunk_size 每个Chunk最大字符数
    overlap    相邻Chunk重叠字符数
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# 5. 读取简历
# ============================================================

def load_resumes():
    """
    从 data/resumes 加载所有 TXT 简历
    """

    resumes = []

    if not RESUME_DIR.exists():

        raise FileNotFoundError(
            f"简历目录不存在：{RESUME_DIR}"
        )

    files = list(
        RESUME_DIR.glob("*.txt")
    )

    if not files:

        raise RuntimeError(
            f"没有找到TXT简历：{RESUME_DIR}"
        )

    for index, file_path in enumerate(files, start=1):

        text = file_path.read_text(
            encoding="utf-8"
        )

        candidate_id = (
            f"candidate_{index:03d}"
        )

        candidate_name = (
            file_path.stem
        )

        resumes.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "file_name": file_path.name,
            "text": text
        })

    return resumes


# ============================================================
# 6. 初始化Embedding模型
# ============================================================

def load_embedding_model():

    print("\n正在加载 Embedding 模型...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        f"Embedding模型加载成功："
        f"{EMBEDDING_MODEL_NAME}"
    )

    return model


# ============================================================
# 7. 初始化Rerank模型
# ============================================================

def load_reranker():

    print("\n正在加载 Rerank 模型...")

    model = CrossEncoder(
        RERANK_MODEL_NAME
    )

    print(
        f"Rerank模型加载成功："
        f"{RERANK_MODEL_NAME}"
    )

    return model


# ============================================================
# 8. 初始化ChromaDB
# ============================================================

def create_chroma_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "删除旧的 ChromaDB Collection"
        )

    except Exception:

        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
                "AI Recruitment Candidate Resume Vector Database"
        }
    )

    print(
        f"ChromaDB Collection创建成功："
        f"{COLLECTION_NAME}"
    )

    return collection


# ============================================================
# 9. 简历入库
# ============================================================

def index_resumes(
        collection,
        resumes,
        embedding_model
):

    documents = []

    embeddings = []

    ids = []

    metadatas = []

    print("\n开始处理候选人简历...")

    for resume in resumes:

        candidate_id = (
            resume["candidate_id"]
        )

        candidate_name = (
            resume["candidate_name"]
        )

        text = resume["text"]

        chunks = chunk_text(text)

        print(
            f"\n候选人：{candidate_name}"
        )

        print(
            f"Chunk数量：{len(chunks)}"
        )

        for chunk_index, chunk in enumerate(chunks):

            vector = embedding_model.encode(
                chunk,
                normalize_embeddings=True
            ).tolist()

            document_id = (
                f"{candidate_id}_chunk_{chunk_index}"
            )

            documents.append(chunk)

            embeddings.append(vector)

            ids.append(document_id)

            metadatas.append({
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "file_name": resume["file_name"],
                "chunk_index": chunk_index
            })

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print("\n================================")

    print(
        f"简历向量入库完成"
    )

    print(
        f"候选人数：{len(resumes)}"
    )

    print(
        f"Chunk总数：{len(documents)}"
    )

    print("================================")


# ============================================================
# 10. ChromaDB向量检索
# ============================================================

def vector_search(
        collection,
        embedding_model,
        jd,
        top_k=VECTOR_TOP_K
):

    print("\n开始JD向量检索...")

    jd_embedding = embedding_model.encode(
        jd,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[
            jd_embedding
        ],
        n_results=min(
            top_k,
            collection.count()
        ),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    search_results = []

    documents = result["documents"][0]

    metadatas = result["metadatas"][0]

    distances = result["distances"][0]

    for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
    ):

        search_results.append({
            "document": document,
            "metadata": metadata,
            "distance": distance
        })

    print(
        f"向量检索完成："
        f"Top {len(search_results)}"
    )

    return search_results


# ============================================================
# 11. CrossEncoder Rerank
# ============================================================

def rerank_results(
        reranker,
        jd,
        search_results,
        top_k=RERANK_TOP_K
):

    print("\n开始 CrossEncoder Rerank...")

    pairs = []

    for item in search_results:

        pairs.append([
            jd,
            item["document"]
        ])

    scores = reranker.predict(
        pairs
    )

    reranked_results = []

    for item, score in zip(
            search_results,
            scores
    ):

        reranked_results.append({
            **item,
            "rerank_score": float(score)
        })

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    reranked_results = (
        reranked_results[:top_k]
    )

    print(
        f"Rerank完成："
        f"Top {len(reranked_results)}"
    )

    return reranked_results


# ============================================================
# 12. 按候选人聚合
# ============================================================

def aggregate_candidates(
        reranked_results
):

    candidate_map = defaultdict(
        lambda: {
            "candidate_id": None,
            "candidate_name": None,
            "file_name": None,
            "chunks": [],
            "scores": []
        }
    )

    for item in reranked_results:

        metadata = item["metadata"]

        candidate_id = (
            metadata["candidate_id"]
        )

        candidate = (
            candidate_map[candidate_id]
        )

        candidate[
            "candidate_id"
        ] = candidate_id

        candidate[
            "candidate_name"
        ] = metadata["candidate_name"]

        candidate[
            "file_name"
        ] = metadata["file_name"]

        candidate["chunks"].append(
            item["document"]
        )

        candidate["scores"].append(
            item["rerank_score"]
        )

    candidates = []

    for candidate in candidate_map.values():

        scores = candidate["scores"]

        candidate["rerank_score"] = (
            max(scores)
        )

        candidate["average_score"] = (
            sum(scores) / len(scores)
        )

        candidates.append(candidate)

    candidates.sort(
        key=lambda x: (
            x["rerank_score"],
            x["average_score"]
        ),
        reverse=True
    )

    return candidates


# ============================================================
# 13. 打印候选人排序
# ============================================================

def print_candidate_ranking(
        candidates
):

    print("\n")
    print("=" * 70)

    print(
        "候选人 Rerank 排名"
    )

    print("=" * 70)

    for rank, candidate in enumerate(
            candidates,
            start=1
    ):

        print(
            f"\n第 {rank} 名"
        )

        print(
            f"候选人："
            f"{candidate['candidate_name']}"
        )

        print(
            f"最高Rerank分数："
            f"{candidate['rerank_score']:.4f}"
        )

        print(
            f"平均Rerank分数："
            f"{candidate['average_score']:.4f}"
        )

        print(
            f"命中Chunk数量："
            f"{len(candidate['chunks'])}"
        )


# ============================================================
# 14. Gemini LLM评分
# ============================================================

def score_candidate_with_llm(
        candidate,
        jd
):

    if not GEMINI_API_KEY:

        return {
            "error":
                "未配置GEMINI_API_KEY"
        }

    if genai is None:

        return {
            "error":
                "google-genai未正确安装"
        }

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    resume_context = "\n\n".join(
        candidate["chunks"]
    )

    prompt = f"""
你是一名资深技术招聘专家。

请根据岗位JD和候选人简历，
对候选人进行专业匹配分析。

岗位JD：

{jd}

候选人：

姓名：
{candidate["candidate_name"]}

候选人简历相关内容：

{resume_context}

请严格返回JSON。

不要返回Markdown。
不要使用```json。

JSON格式必须如下：

{{
    "candidate_name": "候选人姓名",
    "total_score": 0,
    "skill_score": 0,
    "experience_score": 0,
    "project_score": 0,
    "advantages": [
        "优势1",
        "优势2"
    ],
    "risks": [
        "风险1"
    ],
    "missing_skills": [
        "缺失技能1"
    ],
    "recommendation": "强烈推荐/推荐/建议面试/不推荐",
    "analysis": "简短专业分析"
}}

评分范围：

total_score：0-100
skill_score：0-100
experience_score：0-100
project_score：0-100
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        response_text = (
            response.text.strip()
        )

        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(
            response_text
        )

    except Exception as e:

        return {
            "candidate_name":
                candidate["candidate_name"],

            "error":
                str(e)
        }


# ============================================================
# 15. LLM评分
# ============================================================

def llm_scoring(
        candidates,
        jd,
        top_k=CANDIDATE_TOP_K
):

    print("\n")
    print("=" * 70)

    print(
        "开始 LLM 候选人评分"
    )

    print("=" * 70)

    results = []

    for candidate in candidates[:top_k]:

        print(
            f"\n正在分析："
            f"{candidate['candidate_name']}"
        )

        result = score_candidate_with_llm(
            candidate,
            jd
        )

        results.append(result)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=4
            )
        )

    return results


# ============================================================
# 16. 主程序
# ============================================================

def main():

    print("\n")
    print("=" * 70)

    print(
        "AI Recruitment RAG System"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Step 1
    # 加载简历
    # --------------------------------------------------------

    resumes = load_resumes()

    print(
        f"\n加载简历成功："
        f"{len(resumes)} 人"
    )


    # --------------------------------------------------------
    # Step 2
    # 加载Embedding模型
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )


    # --------------------------------------------------------
    # Step 3
    # 加载Reranker
    # --------------------------------------------------------

    reranker = (
        load_reranker()
    )


    # --------------------------------------------------------
    # Step 4
    # 创建ChromaDB
    # --------------------------------------------------------

    collection = (
        create_chroma_collection()
    )


    # --------------------------------------------------------
    # Step 5
    # 简历向量化并入库
    # --------------------------------------------------------

    index_resumes(
        collection=collection,
        resumes=resumes,
        embedding_model=embedding_model
    )


    # --------------------------------------------------------
    # Step 6
    # JD向量检索
    # --------------------------------------------------------

    search_results = vector_search(
        collection=collection,
        embedding_model=embedding_model,
        jd=JD,
        top_k=VECTOR_TOP_K
    )


    # --------------------------------------------------------
    # Step 7
    # Rerank
    # --------------------------------------------------------

    reranked_results = rerank_results(
        reranker=reranker,
        jd=JD,
        search_results=search_results,
        top_k=RERANK_TOP_K
    )


    # --------------------------------------------------------
    # Step 8
    # 按候选人聚合
    # --------------------------------------------------------

    candidates = aggregate_candidates(
        reranked_results
    )


    # --------------------------------------------------------
    # Step 9
    # 输出Rerank排名
    # --------------------------------------------------------

    print_candidate_ranking(
        candidates
    )


    # --------------------------------------------------------
    # Step 10
    # LLM评分
    # --------------------------------------------------------

    llm_results = llm_scoring(
        candidates=candidates,
        jd=JD,
        top_k=CANDIDATE_TOP_K
    )


    # --------------------------------------------------------
    # Step 11
    # 最终排序
    # --------------------------------------------------------

    valid_results = [
        item
        for item in llm_results
        if "total_score" in item
    ]

    valid_results.sort(
        key=lambda x: x["total_score"],
        reverse=True
    )


    print("\n")
    print("=" * 70)

    print(
        "最终候选人排名"
    )

    print("=" * 70)

    for rank, result in enumerate(
            valid_results,
            start=1
    ):

        print(
            f"\n第 {rank} 名"
        )

        print(
            f"候选人："
            f"{result['candidate_name']}"
        )

        print(
            f"综合评分："
            f"{result['total_score']}"
        )

        print(
            f"技术匹配："
            f"{result['skill_score']}"
        )

        print(
            f"经验匹配："
            f"{result['experience_score']}"
        )

        print(
            f"项目匹配："
            f"{result['project_score']}"
        )

        print(
            f"建议："
            f"{result['recommendation']}"
        )

    print("\n")
    print("=" * 70)

    print(
        "AI Recruitment RAG System 执行完成"
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()