"""
文档预处理管线
加载文档 → 切块 → 向量化 → 存入 ChromaDB
"""
import os
import glob
import chromadb
from embedding import OpenVINOEmbeddingFunction


def load_documents(data_dir: str) -> list[dict]:
    """
    加载目录下的所有文本文件

    Args:
        data_dir: 文档目录路径
    Returns:
        [{"content": str, "source": str}, ...]
    """
    documents = []
    patterns = ["**/*.txt", "**/*.md", "**/*.py"]

    for pattern in patterns:
        for filepath in glob.glob(os.path.join(data_dir, pattern), recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    documents.append({
                        "content": content,
                        "source": os.path.relpath(filepath, data_dir),
                    })
            except Exception as e:
                print(f"⚠️ 跳过 {filepath}: {e}")

    return documents


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    按字符数切块（中文友好）

    Args:
        text: 原始文本
        chunk_size: 每块最大字符数
        overlap: 块间重叠字符数
    Returns:
        文本块列表
    """
    # 先按段落分割
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果当前块加上新段落不超限，合并
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk)
            # 如果单个段落就超限，按句子切分
            if len(para) > chunk_size:
                sentences = para.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
                sub_chunk = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(sub_chunk) + len(sent) + 1 <= chunk_size:
                        sub_chunk = sub_chunk + sent if sub_chunk else sent
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # 添加重叠
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # 从前一块末尾取 overlap 字符
            prev_tail = chunks[i - 1][-overlap:]
            overlapped_chunks.append(prev_tail + chunks[i])
        chunks = overlapped_chunks

    return chunks


def ingest(
    data_dir: str = None,
    db_path: str = None,
    embedding_model_path: str = None,
    embedding_device: str = "CPU",
    collection_name: str = "documents",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """
    执行文档预处理管线

    Args:
        data_dir: 文档目录
        db_path: ChromaDB 持久化路径
        embedding_model_path: Embedding 模型路径
        embedding_device: Embedding 推理设备
        collection_name: ChromaDB 集合名称
        chunk_size: 切块大小
        chunk_overlap: 块间重叠
    """
    base_dir = os.path.dirname(__file__)
    data_dir = data_dir or os.path.join(base_dir, "data", "docs")
    db_path = db_path or os.path.join(base_dir, "db")
    embedding_model_path = embedding_model_path or os.path.join(base_dir, "models", "qwen3-embedding-0.6b-int4")

    print("=" * 50)
    print("📄 文档预处理管线")
    print("=" * 50)

    # 1. 加载文档
    print(f"\n📂 加载文档: {data_dir}")
    documents = load_documents(data_dir)
    if not documents:
        print("❌ 未找到任何文档，请将 .txt / .md 文件放入 data/docs/ 目录")
        return
    print(f"   找到 {len(documents)} 个文档")

    # 2. 切块
    print(f"\n✂️ 切块 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    all_chunks = []
    all_ids = []
    all_metadatas = []
    for doc in documents:
        chunks = split_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['source']}_{i}")
            all_metadatas.append({"source": doc["source"], "chunk_index": i})
    print(f"   切成 {len(all_chunks)} 个文本块")

    # 3. 初始化 Embedding 和 ChromaDB
    print(f"\n🔧 初始化 Embedding 模型 (设备: {embedding_device})")
    embedding_fn = OpenVINOEmbeddingFunction(
        model_path=embedding_model_path,
        device=embedding_device,
    )

    print(f"\n💾 存入 ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # 删除旧集合（如果存在）
    try:
        client.delete_collection(collection_name)
        print(f"   已删除旧集合 '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    # 4. 批量写入
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_metas = all_metadatas[i : i + batch_size]
        collection.add(
            documents=batch_chunks,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        print(f"   已写入 {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} 块")

    print(f"\n{'='*50}")
    print(f"✅ 预处理完成！")
    print(f"   集合: {collection_name}")
    print(f"   文档数: {len(documents)}")
    print(f"   文本块数: {len(all_chunks)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文档预处理管线")
    parser.add_argument("--data-dir", type=str, help="文档目录路径")
    parser.add_argument("--db-path", type=str, help="ChromaDB 持久化路径")
    parser.add_argument("--model-path", type=str, help="Embedding 模型路径")
    parser.add_argument("--device", type=str, default="CPU", help="推理设备")
    parser.add_argument("--collection", type=str, default="documents", help="集合名称")
    parser.add_argument("--chunk-size", type=int, default=512, help="切块大小")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="块间重叠")

    args = parser.parse_args()
    ingest(
        data_dir=args.data_dir,
        db_path=args.db_path,
        embedding_model_path=args.model_path,
        embedding_device=args.device,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
