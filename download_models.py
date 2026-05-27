"""
模型下载脚本
从 ModelScope 下载三个 OpenVINO 格式模型到本地
"""
from modelscope import snapshot_download
import os

# ModelScope 上的 OpenVINO 组织模型
MODELS = {
    "qwen3-4b-int4": "OpenVINO/Qwen3-4B-int4-ov",
    "qwen3-embedding-0.6b-int4": "OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov",
    "qwen3-reranker-0.6b-int8": "OpenVINO/Qwen3-Reranker-0.6B-int8-ov",
}

def download_all():
    base_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(base_dir, exist_ok=True)

    for name, repo_id in MODELS.items():
        local_dir = os.path.join(base_dir, name)
        print(f"\n{'='*50}")
        print(f"下载: {repo_id}")
        print(f"目标: {local_dir}")
        print(f"{'='*50}")
        try:
            snapshot_download(
                model_id=repo_id,
                local_dir=local_dir,
            )
            print(f"✅ {name} 下载完成")
        except Exception as e:
            print(f"❌ {name} 下载失败: {e}")
            print(f"   请手动从 https://www.modelscope.cn/models/{repo_id} 下载")

    print(f"\n{'='*50}")
    print("全部下载完成！")
    print(f"{'='*50}")

if __name__ == "__main__":
    download_all()
