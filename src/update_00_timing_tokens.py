import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\00_cloud_lab_setup.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update Cell 252+ with new generation code that includes timing and larger token limit
cell_source = [
    'import time\n',
    '\n',
    'messages = [\n',
    '    {"role": "system", "content": "你是一名经验丰富的大语言模型架构师，说话风格硬核幽默，擅长用大白话讲解技术。"},\n',
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
    '# 因为最新的模型可能会输出大量的 <thinking> 思维链过程，我们将生成的 token 数量限制上限调大到 2048\n',
    'start_time = time.time()\n',
    '\n',
    '# 生成回答！\n',
    'outputs = model.generate(\n',
    '    **inputs,\n',
    '    max_new_tokens=2048,\n',
    '    temperature=0.7\n',
    ')\n',
    '\n',
    'end_time = time.time()\n',
    '\n',
    '# 提取核心亮点：利用 inputs["input_ids"].shape[-1]: 优雅地切片，只保留模型生成的文字结果\n',
    'generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]\n',
    'response = tokenizer.decode(generated_tokens, skip_special_tokens=True)\n',
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
data['cells'][16]['source'] = cell_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
