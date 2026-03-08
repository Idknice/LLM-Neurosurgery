import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\00_cloud_lab_setup.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update Cell 12: Download logic with cache check
cell_12_source = [
    'from huggingface_hub import snapshot_download\n',
    'import time\n',
    'import os\n',
    '\n',
    '# 我们设定要下载的模型 ID\n',
    'MODEL_ID = "Qwen/Qwen3.5-4B"\n',
    '\n',
    '# 简单的防重复下载检查\n',
    '# HuggingFace 的缓存目录默认采用 models--作者--模型名 的结构\n',
    'model_cache_path = os.path.join(CACHE_DIR, f"models--{MODEL_ID.replace(\'/\', \'--\')}")\n',
    'if os.path.exists(model_cache_path) and len(os.listdir(model_cache_path)) > 0:\n',
    '    print(f"🎉 发现本地缓存：{model_cache_path}\\n✅ 看来你之前已经下载过该模型了，本次执行将采取极速核对模式！\\n")\n',
    'else:\n',
    '    print(f"🚀 开始全新同步模型 [{MODEL_ID}] 到持久化目录...\\n")\n',
    '\n',
    'print(f"存放路径: {CACHE_DIR}")\n',
    'print("⏳ 这可能需要 2-5 分钟，喝口水耐心等待。如果之前已经下载过，瞬间就会完成！")\n',
    '\n',
    'start_time = time.time()\n',
    '\n',
    '# snapshot_download 会非常聪明地处理断点续传和多线程下载\n',
    'model_path = snapshot_download(\n',
    '    repo_id=MODEL_ID, \n',
    '    cache_dir=CACHE_DIR,\n',
    '    # 排除一些我们不需要在推理时下载的巨大格式文件 (如 GGUF, 有的时候包含在官方仓库里)\n',
    '    ignore_patterns=["*.gguf", "*.pt", "*.ckpt"]\n',
    ')\n',
    '\n',
    'end_time = time.time()\n',
    'print(f"\\n🎉 模型同步完成! 用时: {end_time - start_time:.2f} 秒")\n',
    'print(f"模型实际物理路径: {model_path}")'
]
data['cells'][12]['source'] = cell_12_source

# Update Cell 16: Generation logic with modern apply_chat_template syntax
cell_16_source = [
    'messages = [\n',
    '    {"role": "system", "content": "你是一名经验丰富的大语言模型架构师，说话风格硬核幽针，擅长用大白话讲解技术。"},\n',
    '    {"role": "user", "content": "你好，我是一名小白，决定今天要亲手解剖你的 Transformer 架构。能给我几条忠告吗？"}\n',
    ']\n',
    '\n',
    '# 借助最新的 apply_chat_template，一行代码完成 Prompt 拼接、返回张量和转移至 GPU 设备\n',
    'inputs = tokenizer.apply_chat_template(\n',
    '    messages,\n',
    '    add_generation_prompt=True,\n',
    '    tokenize=True,\n',
    '    return_dict=True,\n',
    '    return_tensors="pt"\n',
    ').to(model.device)\n',
    '\n',
    'print("\\n================ 🤖 Qwen3.5 正在 GPU 上吟唱 =================\\n")\n',
    '\n',
    '# 生成回答！限制最多吐出 512 个字\n',
    'outputs = model.generate(\n',
    '    **inputs,\n',
    '    max_new_tokens=512,\n',
    '    temperature=0.7\n',
    ')\n',
    '\n',
    '# 提取核心亮点：利用 inputs["input_ids"].shape[-1]: 优雅地切片，只保留模型生成的文字结果\n',
    'response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)\n',
    'print(response)'
]
data['cells'][16]['source'] = cell_16_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
