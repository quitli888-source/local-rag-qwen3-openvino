"""
Gradio Web UI
文档上传 + 问答对话 + 检索来源展示
"""
import os
import gradio as gr
from ingest import ingest, load_documents, split_text
from rag import LocalRAG


# ─── 全局配置 ────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "db")
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")

# 默认设备配置
LLM_DEVICE = "GPU"
EMBEDDING_DEVICE = "CPU"
RERANKER_DEVICE = "CPU"


# ─── RAG 系统初始化 ──────────────────────────────────────
rag_system = None


def init_rag():
    global rag_system
    if rag_system is None:
        rag_system = LocalRAG(
            llm_model_path=os.path.join(MODELS_DIR, "qwen3-4b-int4"),
            embedding_model_path=os.path.join(MODELS_DIR, "qwen3-embedding-0.6b-int4"),
            reranker_model_path=os.path.join(MODELS_DIR, "qwen3-reranker-0.6b-int8"),
            db_path=DB_PATH,
            llm_device=LLM_DEVICE,
            embedding_device=EMBEDDING_DEVICE,
            reranker_device=RERANKER_DEVICE,
        )
    return rag_system


# ─── 文档上传处理 ─────────────────────────────────────────
def upload_and_ingest(files):
    """上传文档并执行预处理"""
    if not files:
        return "❌ 请先选择文件"

    os.makedirs(DATA_DIR, exist_ok=True)
    uploaded = []
    for file in files:
        filename = os.path.basename(file.name)
        dest = os.path.join(DATA_DIR, filename)
        with open(file.name, "r", encoding="utf-8") as src:
            content = src.read()
        with open(dest, "w", encoding="utf-8") as dst:
            dst.write(content)
        uploaded.append(filename)

    # 执行预处理
    try:
        ingest(
            data_dir=DATA_DIR,
            db_path=DB_PATH,
            embedding_model_path=os.path.join(MODELS_DIR, "qwen3-embedding-0.6b-int4"),
            embedding_device=EMBEDDING_DEVICE,
        )
        global rag_system
        rag_system = None  # 重置，下次查询时重新加载
        return f"✅ 已上传并处理 {len(uploaded)} 个文件:\n" + "\n".join(f"  • {f}" for f in uploaded)
    except Exception as e:
        return f"❌ 处理失败: {e}"


# ─── RAG 查询 ────────────────────────────────────────────
def rag_query(question, history):
    """执行 RAG 查询"""
    if not question.strip():
        return history

    try:
        rag = init_rag()
        result = rag.query(question, stream=False)
        answer = result["answer"]

        # 构造来源信息
        sources_text = "\n\n---\n📚 **检索来源:**\n"
        for src in result["sources"]:
            sources_text += f"\n**[{src['rank']}]** (相关度: {src['score']})\n{src['content']}\n"

        history.append([question, answer + sources_text])
        return history
    except Exception as e:
        history.append([question, f"❌ 查询失败: {e}"])
        return history


# ─── Gradio 界面 ─────────────────────────────────────────
def create_ui():
    with gr.Blocks(title="本地 RAG 系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 本地 RAG 系统")
        gr.Markdown("基于 OpenVINO + Qwen3 全家桶的本地检索增强生成系统")

        with gr.Row():
            # 左侧：文档管理
            with gr.Column(scale=1):
                gr.Markdown("### 📂 文档管理")
                file_upload = gr.File(
                    label="上传文档 (.txt / .md)",
                    file_count="multiple",
                    file_types=[".txt", ".md"],
                )
                upload_btn = gr.Button("上传并处理", variant="primary")
                upload_status = gr.Textbox(label="状态", lines=5, interactive=False)

                gr.Markdown("### ⚙️ 设备配置")
                gr.Markdown(f"""
                | 组件 | 设备 |
                |------|------|
                | LLM | {LLM_DEVICE} |
                | Embedding | {EMBEDDING_DEVICE} |
                | Reranker | {RERANKER_DEVICE} |
                """)

            # 右侧：问答对话
            with gr.Column(scale=2):
                gr.Markdown("### 💬 问答对话")
                chatbot = gr.Chatbot(height=500)
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入你的问题...",
                    lines=2,
                )
                with gr.Row():
                    query_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

        # 事件绑定
        upload_btn.click(
            fn=upload_and_ingest,
            inputs=[file_upload],
            outputs=[upload_status],
        )
        query_btn.click(
            fn=rag_query,
            inputs=[question_input, chatbot],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[question_input])
        question_input.submit(
            fn=rag_query,
            inputs=[question_input, chatbot],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[question_input])
        clear_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
