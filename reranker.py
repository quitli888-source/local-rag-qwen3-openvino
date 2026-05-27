"""
Reranker 模块
基于 OpenVINO + Qwen3-Reranker 实现文档重排序
使用 openvino runtime 直接推理，绕过 optimum 的参数过滤
"""
import os
import numpy as np
from transformers import AutoTokenizer
import openvino as ov


class DocumentReranker:
    """Qwen3-Reranker 文档重排序器"""

    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Args:
            model_path: OpenVINO 模型目录路径
            device: 推理设备 ("CPU", "GPU", "NPU")
        """
        # 直接用 OpenVINO runtime 加载模型
        core = ov.Core()
        model_xml = os.path.join(model_path, "openvino_model.xml")
        model_bin = os.path.join(model_path, "openvino_model.bin")
        model = core.read_model(model_xml, model_bin)
        self.model = core.compile_model(model, device)
        self.infer_request = self.model.create_infer_request()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left"
        )
        self.max_length = 8192
        # Qwen3-Reranker 使用 yes/no token 的 logits 计算相关性分数
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # 打印模型输入信息（调试用）
        print(f"   Reranker 模型输入: {[p.get_any_name() for p in self.model.inputs]}")

    def compute_score(self, pairs: list[str]) -> list[float]:
        """
        计算 query-document 对的相关性分数

        Args:
            pairs: 格式化的 query-document 对列表
        Returns:
            相关性分数列表 (0~1)
        """
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        # 生成 position_ids
        position_ids = np.cumsum(attention_mask, axis=-1) - 1
        position_ids[attention_mask == 0] = 1

        # 直接推理
        result = self.infer_request.infer({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })

        # 取输出（通常是第一个输出）
        output = list(result.values())[0]

        # 取最后一个 token 的隐藏状态
        hidden_states = output[:, -1, :]

        # 计算 yes/no 的相关性分数
        true_logits = hidden_states[:, self.token_true_id]
        false_logits = hidden_states[:, self.token_false_id]

        # softmax
        max_logits = np.maximum(true_logits, false_logits)
        true_exp = np.exp(true_logits - max_logits)
        false_exp = np.exp(false_logits - max_logits)
        scores = (true_exp / (true_exp + false_exp)).tolist()

        return scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        对文档列表进行重排序

        Args:
            query: 用户查询
            documents: 候选文档列表
            instruction: 可选的任务指令
            top_k: 返回前 K 个结果
        Returns:
            [(document, score), ...] 按分数降序排列
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"

        # 构造 query-document 对
        pairs = [
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]

        scores = self.compute_score(pairs)

        # 按分数降序排列
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            doc_scores = doc_scores[:top_k]

        return doc_scores
