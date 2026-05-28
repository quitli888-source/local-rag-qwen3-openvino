"""验证 LLM 是否真的跑在 iGPU 上"""
import openvino as ov
import openvino_genai as ov_genai
import os, time

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "qwen3-4b-int4")

core = ov.Core()
print(f"可用设备: {core.available_devices}")

# 用 openvino runtime 直接加载模型，查看设备
print(f"\n加载模型: {model_path}")
model_xml = os.path.join(model_path, "openvino_model.xml")
if os.path.exists(model_xml):
    model = core.read_model(model_xml)
    print(f"模型输入: {[p.get_any_name() for p in model.inputs]}")

    # 编译到 GPU
    print("\n编译到 GPU（Intel iGPU）...")
    try:
        compiled = core.compile_model(model, "GPU")
        print(f"✅ 编译成功！实际设备: {compiled.get_property('EXECUTION_DEVICES')}")
    except Exception as e:
        print(f"❌ GPU 编译失败: {e}")

    # 编译到 CPU 对比
    print("\n编译到 CPU...")
    try:
        compiled_cpu = core.compile_model(model, "CPU")
        print(f"✅ CPU 编译成功！实际设备: {compiled_cpu.get_property('EXECUTION_DEVICES')}")
    except Exception as e:
        print(f"❌ CPU 编译失败: {e}")
else:
    print("未找到 openvino_model.xml，跳过")
