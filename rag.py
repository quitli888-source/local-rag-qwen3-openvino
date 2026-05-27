"""
RAG 查询核心
检索 → 重排序 → LLM 生成回答
"""
import os
import sys
import chromadb
import openvino_genai as ov_genai
from embedding import OpenVINOEmbeddingFunction, QueryEmbeddingFunction
from reranker import DocumentReranker


class LocalRAG:
    """本地 RAG 系统"""

    def __init__(
        self,
        llm_model_path: str,
        embedding_model_path: str,
        reranker_model_path: str,
        db_path: str,
        llm_device: str = "GPU.1",
        embedding_device: str = "CPU",
        reranker_device: str = "CPU",
        collection_name: str = "documents",
    ):
        base_dir = os.path.dirname(__file__)

        # 初始化 LLM
        print(f"🔧 加载 LLM: {llm_model_path} → {llm_device}")
        self.llm = ov_genai.LLMPipeline(llm_model_path, llm_device)
        self.gen_config = ov_genai.GenerationConfig()
        self.gen_config.max_new_tokens = 1024
        self.gen_config.temperature = 0.7
        self.gen_config.top_p = 0.9

        # 初始化 Embedding
        print(f"🔧 加载 Embedding: {embedding_model_path} → {embedding_device}")
        self.embedding_fn = OpenVINOEmbeddingFunction(
            model_path=embedding_model_path,
            device=embedding_device,
        )
        self.query_embedder = QueryEmbeddingFunction(
            model_path=embedding_model_path,
            device=embedding_device,
        )

        # 初始化 Reranker
        print(f"🔧 加载 Reranker: {reranker_model_path} → {reranker_device}")
        self.reranker = DocumentReranker(
            model_path=reranker_model_path,
            device=reranker_device,
        )

        # 连接 ChromaDB
        print(f"🔧 连接 ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        print(f"✅ RAG 系统就绪 (文档数: {self.collection.count()})")

    def query(
        self,
        question: str,
        retrieve_k: int = 10,
        rerank_k: int = 3,
        stream: bool = False,
    ) -> dict:
        """
        执行 RAG 查询

        Args:
            question: 用户问题
            retrieve_k: 初始检索数量
            rerank_k: 重排序后保留数量
            stream: 是否流式输出
        Returns:
            {"question": str, "answer": str, "sources": list}
        """
        # 1. 向量检索
        print(f"\n🔍 检索 Top-{retrieve_k}...")
        results = self.collection.query(
            query_texts=[question],
            n_results=retrieve_k,
        )
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        print(f"   检索到 {len(retrieved_docs)} 个候选块")

        # 2. 重排序
        print(f"📊 重排序 → Top-{rerank_k}...")
        reranked = self.reranker.rerank(question, retrieved_docs, top_k=rerank_k)
        reranked_docs = [doc for doc, score in reranked]
        reranked_scores = [score for doc, score in reranked]

        for i, (doc, score) in enumerate(reranked):
            preview = doc[:80].replace("\n", " ")
            print(f"   [{i+1}] ({score:.3f}) {preview}...")

        # 3. 构造 Prompt
        context = "\n\n---\n\n".join(reranked_docs)
        prompt = f"""基于以下参考资料回答用户问题。如果参考资料中没有相关信息，请如实说明。

参考资料：
{context}

用户问题：{question}

请提供准确、详细的回答："""

        # 4. LLM 生成
        print(f"\n💬 生成回答...")
        if stream:
            print("回答: ", end="", flush=True)
            def streamer(subword):
                print(subword, end="", flush=True)
                sys.stdout.flush()
                return False
            self.llm.generate(prompt, self.gen_config, streamer=streamer)
            print()
            answer = "(流式输出已打印)"
        else:
            answer = self.llm.generate(prompt, self.gen_config)

        # 5. 整理来源信息
        sources = []
        for i, (doc, score) in enumerate(reranked):
            sources.append({
                "rank": i + 1,
                "score": round(score, 4),
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


def main():
    """命令行交互模式"""
    import argparse

    parser = argparse.ArgumentParser(description="本地 RAG 查询")
    parser.add_argument("--llm", type=str, help="LLM 模型路径")
    parser.add_argument("--embedding", type=str, help="Embedding 模型路径")
    parser.add_argument("--reranker", type=str, help="Reranker 模型路径")
    parser.add_argument("--db", type=str, help="ChromaDB 路径")
    parser.add_argument("--llm-device", type=str, default="GPU")
    parser.add_argument("--embedding-device", type=str, default="CPU")
    parser.add_argument("--reranker-device", type=str, default="CPU")
    parser.add_argument("--collection", type=str, default="documents")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    rag = LocalRAG(
        llm_model_path=args.llm or os.path.join(base_dir, "models", "qwen3-4b-int4"),
        embedding_model_path=args.embedding or os.path.join(base_dir, "models", "qwen3-embedding-0.6b-int4"),
        reranker_model_path=args.reranker or os.path.join(base_dir, "models", "qwen3-reranker-0.6b-int8"),
        db_path=args.db or os.path.join(base_dir, "db"),
        llm_device=args.llm_device,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        collection_name=args.collection,
    )

    print(f"\n{'='*50}")
    print("本地 RAG 系统已就绪，输入问题开始对话（输入 'quit' 退出）")
    print(f"{'='*50}")

    while True:
        try:
            question = input("\n❓ 你的问题: ").strip()
            if not question or question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            result = rag.query(question, stream=True)
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break


if __name__ == "__main__":
    main()
