import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\00_cloud_lab_setup.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. Update pip install cell
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('!pip install' in line for line in source):
            new_source = [
                '# 使用 -q (quiet) 静默安装，-U (upgrade) 更新到最新版本\n',
                '!pip install -q -U torch transformers accelerate datasets huggingface_hub bitsandbytes qwen-vl-utils flash-linear-attention causal-conv1d git+https://github.com/huggingface/transformers.git\n',
                '\n',
                'import torch\n',
                'import transformers\n',
                '\n',
                'print(f"PyTorch 版本: {torch.__version__}")\n',
                'print(f"Transformers 版本: {transformers.__version__}")\n',
                'print(f"CUDA 是否可用: {torch.cuda.is_available()}")'
            ]
            cell['source'] = new_source

# 2. Add local copy cell after download cell and before load cell
insert_idx = -1
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code':
        if 'AutoModelForImageTextToText' in "".join(cell['source']):
            insert_idx = i
            break

if insert_idx != -1:
    local_copy_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "drive_cache_dir = \"/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache/models--Qwen--Qwen3.5-4B\"\n",
            "local_model_path = \"/content/qwen3_5_local\"\n",
            "\n",
            "# 极其重要的高速化步骤：将网盘数据瞬间拉取到本机的系统高速固态硬盘里！\n",
            "if not os.path.exists(local_model_path):\n",
            "    print(\"🚗 正在将几十 GB 的模型权重从网盘大动脉抽血到 Colab 本地 SSD 高速通道...\")\n",
            "    print(\"⏳ 这个复制操作因为是内网传输，只要一两分钟，但这绝对能让你接下来的加载和实验起飞！\")\n",
            "    !cp -r {drive_cache_dir} {local_model_path}\n",
            "else:\n",
            "    print(\"⚡ 本地高速缓存库已就绪！可以开始呼叫外星人了！\")"
        ]
    }
    data['cells'].insert(insert_idx, local_copy_cell)

# 3. Update load cell to use local path
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
                'model = AutoModelForImageTextToText.from_pretrained(\n',
                '    local_model_path,\n',
                '    torch_dtype="auto",\n',
                '    device_map="auto",\n',
                '    trust_remote_code=True\n',
                ')\n',
                '\n',
                'print("✅ 模型加载成功！显存已经全部就绪！")'
            ]
            cell['source'] = new_source

# 4. Rollback chat iteration to text
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('apply_chat_template' in line for line in source):
            new_source = [
                'import time\n',
                '\n',
                'messages = [\n',
                '    {"role": "system", "content": "你是一名经验丰富的大语言模型极客导师，说话风格硬核幽默，擅长用大白话讲解技术。"},\n',
                '    {"role": "user", "content": "你好，我是一名小白，决定今天要亲手解剖你的 Transformer 架构。能给我几条忠告吗？"}\n',
                ']\n',
                '\n',
                '# 使用 Processor 打包纯文本消息（即便没有图片，VLM 的 Processor 也能完美兼容纯文本流程）\n',
                'inputs = processor.apply_chat_template(\n',
                '    messages,\n',
                '    add_generation_prompt=True,\n',
                '    tokenize=True,\n',
                '    return_dict=True,\n',
                '    return_tensors="pt"\n',
                ').to(model.device)\n',
                '\n',
                'print("\\n================ 🤖 Qwen3.5 正在 GPU 上凝视你的请求并吟唱 =================\\n")\n',
                '\n',
                'start_time = time.time()\n',
                '\n',
                '# 生成回答（文本生成很快，所以我们把 token 数量放宽一点！）\n',
                'outputs = model.generate(\n',
                '    **inputs,\n',
                '    max_new_tokens=2048\n',
                ')\n',
                '\n',
                'end_time = time.time()\n',
                '\n',
                'generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]\n',
                'response = processor.decode(generated_tokens, skip_special_tokens=True)\n',
                '\n',
                'print(response)\n',
                '\n',
                '# ====== 性能统计 ======\n',
                'generation_time = end_time - start_time\n',
                'token_count = len(generated_tokens)\n',
                'tokens_per_second = token_count / generation_time\n',
                '\n',
                'print("\\n================ 📊 生成性能统计 =================")\n',
                'print(f"⏱️ 耗时: {generation_time:.2f} 秒")\n',
                'print(f"📝 生成的 Token 数: {token_count}")\n',
                'print(f"🚀 生成速度: {tokens_per_second:.2f} tokens/s")'
            ]
            cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
    f.write("\n")
