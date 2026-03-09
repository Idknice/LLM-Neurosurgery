import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\00_cloud_lab_setup.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update cell containing pip install
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('!pip install' in line for line in source):
            new_source = [
                '# 使用 -q (quiet) 静默安装，-U (upgrade) 更新到最新版本\n',
                '# 注意：为了让小白快速体验，我们移除了耗时极长的最新源码编译和底层 C++ 加速算子\n',
                '!pip install -q -U torch transformers accelerate datasets huggingface_hub\n',
                '\n',
                'import torch\n',
                'import transformers\n',
                '\n',
                'print(f"PyTorch 版本: {torch.__version__}")\n',
                'print(f"Transformers 版本: {transformers.__version__}")\n',
                'print(f"CUDA 是否可用: {torch.cuda.is_available()}")'
            ]
            cell['source'] = new_source

# Because we removed bitsandbytes and qwen-vl-utils, we might need to make sure 
# the 00 notebook uses standard fp16/bf16 loading instead of requiring specialized libraries
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('AutoModelForImageTextToText.from_pretrained' in line for line in source):
            new_source = [
                'from transformers import AutoProcessor, AutoModelForImageTextToText\n',
                'import torch\n',
                '\n',
                'print("🧠 正在将模型从本地 SSD 高速通道拔出，装配进 GPU 显存中...")\n',
                '\n',
                'local_model_path = "/content/qwen3_5_local"\n',
                '\n',
                '# 使用本地缓存路径高速加载\n',
                'processor = AutoProcessor.from_pretrained(local_model_path)\n',
                '\n',
                '# 采用 bfloat16 半精度加载（T4 原生支持，极大节省完整权重的显存占用）\n',
                'model = AutoModelForImageTextToText.from_pretrained(\n',
                '    local_model_path,\n',
                '    torch_dtype=torch.bfloat16,\n',
                '    device_map="auto"\n',
                ')\n',
                '\n',
                'print("✅ 模型加载成功！显存已经全部就绪！")'
            ]
            cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
    f.write("\n")
