# 本地 RAG 系统（OpenVINO + Qwen3）

基于 Intel 酷睿 Ultra 处理器的本地检索增强生成（RAG）系统。所有数据在本地处理，无需上传云端。

## 硬件环境

| 组件 | 型号 | 实际用途 |
|------|------|---------|
| CPU | Intel Core Ultra 9 275HX（24核） | Embedding / Reranker 推理 + 数据处理 |
| iGPU | Intel Graphics（GPU.0） | LLM（Qwen3-4B）推理加速 |
| NPU | Intel AI Boost（13 TOPS） | 已识别，因动态维度限制暂未使用 |
| dGPU | NVIDIA RTX 5070 Laptop（8GB） | OpenVINO 不支持 NVIDIA GPU，暂未使用 |
| 内存 | 32GB DDR5 | 模型加载与运行 |

## 架构

```
用户提问
    │
    ▼
文档预处理（离线）          查询流程（在线）
┌─────────────┐          ┌─────────────────┐
│ 文档加载     │          │ 向量检索         │
│ ↓           │          │ ↓               │
│ 切块 512tok │          │ Reranker 重排序  │
│ ↓           │          │ ↓               │
│ Embedding   │          │ Qwen3-4B 生成    │
│ ↓           │          │                 │
│ ChromaDB    │          │ 返回回答+来源    │
└─────────────┘          └─────────────────┘
```

## 模型

| 环节 | 模型 | 格式 | 大小 | 设备 |
|------|------|------|------|------|
| 向量化 | Qwen3-Embedding-0.6B | OpenVINO INT4 | 426 MB | CPU |
| 重排序 | Qwen3-Reranker-0.6B | OpenVINO INT8 | 624 MB | CPU |
| 生成 | Qwen3-4B | OpenVINO INT4 | 2.26 GB | Intel iGPU（GPU.0） |

## 快速开始

### 1. 环境准备

```bash
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

### 2. 下载模型

```bash
pip install modelscope
python download_models.py
```

模型下载到 `models/` 目录，共约 3.3 GB。

### 3. 文档预处理

将 `.txt` / `.md` 文件放入 `data/docs/` 目录，然后执行：

```bash
python ingest.py
```

### 4. 启动

**命令行模式：**
```bash
python rag.py
```

**Web UI 模式：**
```bash
python app.py
# 浏览器打开 http://localhost:7860
```

## 项目结构

```
├── models/                      # 模型文件（需下载）
├── data/docs/                   # 用户文档
├── db/                          # ChromaDB 持久化
├── requirements.txt             # Python 依赖
├── download_models.py           # 模型下载脚本
├── embedding.py                 # Embedding 模块（OpenVINO + Qwen3）
├── reranker.py                  # Reranker 模块（OpenVINO + Qwen3）
├── ingest.py                    # 文档预处理管线
├── rag.py                       # RAG 查询核心（命令行）
└── app.py                       # Gradio Web UI
```

## 已知问题

- **NPU 不支持动态维度**：OpenVINO 的 NPU 插件要求静态输入形状，而 Qwen3 Embedding/Reranker 模型使用动态维度，无法在 NPU 上运行
- **OpenVINO 不支持 NVIDIA GPU**：OpenVINO 的 OpenCL 后端与 NVIDIA GPU 不兼容，RTX 5070 独显暂未利用
- **改进方向**：通过 Ollama（CUDA）替代 OpenVINO 做 LLM 推理，以利用 RTX 5070 独显获得更高推理速度

## License

MIT
