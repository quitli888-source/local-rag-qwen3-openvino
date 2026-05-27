"""
Embedding 模块
基于 OpenVINO + Qwen3-Embedding 实现文本向量化
兼容 ChromaDB 的 EmbeddingFunction 接口
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from optimum.intel import OVModelForFeatureExtraction
from chromadb import Documents, EmbeddingFunction, Embeddings


class TextEmbedder:
    """Qwen3-Embedding 文本向量化器"""

    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Args:
            model_path: OpenVINO 模型目录路径
            device: 推理设备 ("CPU", "GPU", "NPU")
        """
        self.model = OVModelForFeatureExtraction.from_pretrained(
            model_path, device=device, export=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left"
        )
        self.max_length = 8192
        self.device = device

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """Qwen3 Embedding 专用的池化：取最后一个有效 token 的隐藏状态"""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def embed(self, texts: list[str], task: str | None = None) -> np.ndarray:
        """
        计算文本列表的 embedding 向量

        Args:
            texts: 文本列表
            task: 可选的任务指令（提升检索效果约1-5%）
        Returns:
            numpy 数组, shape=(len(texts), dim)
        """
        if task:
            texts = [f"Instruct: {task}\nQuery:{t}" for t in texts]

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


class OpenVINOEmbeddingFunction(EmbeddingFunction):
    """ChromaDB 兼容的 Embedding Function，内部使用 OpenVINO 推理"""

    def __init__(self, model_path: str, device: str = "CPU", task: str | None = None):
        """
        Args:
            model_path: OpenVINO 模型目录路径
            device: 推理设备
            task: 可选的任务指令
        """
        self.embedder = TextEmbedder(model_path, device)
        self.task = task or "Given a document, retrieve relevant passages"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.embedder.embed(input, task=None)  # 文档不需要指令
        return embeddings.tolist()


class QueryEmbeddingFunction:
    """查询专用 Embedding Function（带指令前缀）"""

    def __init__(self, model_path: str, device: str = "CPU"):
        self.embedder = TextEmbedder(model_path, device)
        self.task = "Given a web search query, retrieve relevant passages that answer the query"

    def __call__(self, query: str) -> list[float]:
        embedding = self.embedder.embed([query], task=self.task)
        return embedding[0].tolist()

    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        embeddings = self.embedder.embed(queries, task=self.task)
        return embeddings.tolist()
