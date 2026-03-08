import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\01_pytorch_huggingface_basics.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update cell containing pip install
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('!pip install' in line for line in source):
            new_source = [
                '# 1. 安装我们在解剖过程中必需的屠龙刀\n',
                '# 除了基础库，还加入了 flash-linear-attention 和 causal-conv1d 以消除量化加载时的 Warning 并极大加速推理\n',
                '!pip install -q -U torch transformers accelerate datasets huggingface_hub bitsandbytes qwen-vl-utils flash-linear-attention causal-conv1d git+https://github.com/huggingface/transformers.git'
            ]
            cell['source'] = new_source

# Update the cache mapping cell to point to local drive and auto-copy
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('drive.mount' in line for line in source):
            new_source = [
                'from google.colab import drive\n',
                'import os\n',
                '\n',
                '# 2. 挂载持久化硬盘\n',
                'drive.mount("/content/drive")\n',
                '\n',
                '# 3. 指定远程 Drive 路径和极速“本地”路径\n',
                '# 注意：这里的“本地”指的是 Colab 平台免费分配给你的临时虚拟机内部的高速 SSD，\n',
                '# 绝对不是要下载到你自己的真实电脑上！\n',
                'drive_cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache/models--Qwen--Qwen3.5-4B"\n',
                'local_model_path = "/content/qwen3_5_local"\n',
                '\n',
                '# 将庞大的破碎碎文件（模型权重）从慢速的网盘 I/O 中解救出来，一次性拉到虚拟机的系统高速 SSD 中！\n',
                'if not os.path.exists(local_model_path):\n',
                '    print("🚗 正在将几十 GB 的模型权重从网盘大动脉抽血到 Colab 本地 SSD 高速通道...")\n',
                '    print("⏳ 这个复制操作因为是内网传输，只要一两分钟，但这绝对能让你把加载几十分钟的时间变为几秒钟！起飞！")\n',
                '    !cp -r {drive_cache_dir} {local_model_path}\n',
                'else:\n',
                '    print("⚡ 本地高速缓存库已就绪！")\n',
                '\n',
                'print("✅ 工作区与高速缓存目录已绑定完毕！")'
            ]
            cell['source'] = new_source

# Replace AutoProcessor load cell
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('processor = AutoProcessor.from_pretrained' in line for line in source):
            new_source = [
                'from transformers import AutoProcessor\n',
                '\n',
                'print("正在加载 Processor (整合了分词器和图像预处理器)...")\n',
                '# 我们直接从刚刚拷贝到虚拟机系统盘的 local_model_path 里面闪电读取\n',
                'processor = AutoProcessor.from_pretrained(local_model_path)\n',
                '\n',
                '# 获取底层的文本分词器看看\n',
                'tokenizer = processor.tokenizer\n',
                'print(f"Qwen3.5 词表大小: {tokenizer.vocab_size} 个词块")'
            ]
            cell['source'] = new_source

# Replace Model 4-bit load cell
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('model_4bit = AutoModelForImageTextToText.from_pretrained' in line for line in source):
            new_source = [
                'from transformers import AutoModelForImageTextToText, BitsAndBytesConfig\n',
                'import torch\n',
                '\n',
                'print("正在使用 4-bit 量化魔法从虚拟机的系统 SSD 闪电加载多模态架构 (Qwen3.5-4B)...")\n',
                'quantization_config = BitsAndBytesConfig(\n',
                '    load_in_4bit=True,\n',
                '    bnb_4bit_compute_dtype=torch.bfloat16,\n',
                '    bnb_4bit_quant_type="nf4"\n',
                ')\n',
                '\n',
                '# 从 local_model_path 高速加载，避免 Google Drive 极其悲惨的网络 I/O 速度\n',
                'model_4bit = AutoModelForImageTextToText.from_pretrained(\n',
                '    local_model_path,\n',
                '    device_map="auto",\n',
                '    quantization_config=quantization_config,\n',
                '    trust_remote_code=True\n',
                ')\n',
                '\n',
                'print("\\n✅ 量化加载完成！这速度是不是像踩了油门一样？！")'
            ]
            cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
    f.write("\n")
