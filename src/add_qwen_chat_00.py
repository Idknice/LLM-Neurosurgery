import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\00_cloud_lab_setup.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 5. 首次唤醒：与 Qwen3.5 的第一次对话\n',
            '\n',
            '既然模型已经乖乖躺在我们永久挂载的硬盘里了，下面我们把它从机械的硬盘调动到最核心的 **GPU 的显存 (VRAM)** 中，并进行一次最基础的对话尝试。\n',
            '\n',
            '这也是本门极客课程的第一个小高潮——彻底验证整个下载、挂载过程没有跑偏。'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'from transformers import AutoModelForCausalLM, AutoTokenizer\n',
            'import torch\n',
            '\n',
            'print("🧠 正在将模型加载到 GPU 显存中，这大概需要几分钟。此时如果开启一个新终端运行 nvidia-smi 能够看到显存占用飙升...")\n',
            '\n',
            '# 之前 snapshot_download 时保存的缓存目录，通过这个机制能够避开重复下载\n',
            'cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache"\n',
            'model_id = "Qwen/Qwen3.5-4B"\n',
            '\n',
            'tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)\n',
            '\n',
            '# device_map="auto" 会让 transformers 自动帮我们把模型放到可用的 GPU 上\n',
            '# torch_dtype="auto" 会自动使用 float16 或 bfloat16 以节约显卡显存\n',
            'model = AutoModelForCausalLM.from_pretrained(\n',
            '    model_id,\n',
            '    cache_dir=cache_dir,\n',
            '    torch_dtype="auto",\n',
            '    device_map="auto"\n',
            ')\n',
            '\n',
            'print("✅ 模型加载成功！显存已经全部就绪！")'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '接下来，我们借助最新的 `apply_chat_template` 工具，给大模型输入一段简单的指令，看看它在我们的 Colab GPU 上会有怎样的本能反应。'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'messages = [\n',
            '    {"role": "system", "content": "你是一名经验丰富的大语言模型架构师，说话风格硬核幽默，擅长用大白话讲解冷核聚变般的深奥技术。"},\n',
            '    {"role": "user", "content": "你好，我是一名小白，决定今天要亲手解剖你的 Transformer 架构。能给我几条忠告吗？"}\n',
            ']\n',
            '\n',
            '# 使用官方推荐的 apply_chat_template 将对话字典组装成 Qwen 能看懂的特殊 prompt 字符串\n',
            'text = tokenizer.apply_chat_template(\n',
            '    messages,\n',
            '    tokenize=False,\n',
            '    add_generation_prompt=True\n',
            ')\n',
            '\n',
            '# 将文本转成数字张量丢给对应的 GPU 设备\n',
            'model_inputs = tokenizer([text], return_tensors="pt").to(model.device)\n',
            '\n',
            'print("\\n================ 🤖 Qwen3.5 正在 GPU 上吟唱 =================\\n")\n',
            '\n',
            '# 生成回答！限制最多吐出 512 个字\n',
            'generated_ids = model.generate(\n',
            '    **model_inputs,\n',
            '    max_new_tokens=512,\n',
            '    temperature=0.7\n',
            ')\n',
            '\n',
            '# 截断输入的部分，只保留模型全新生成的文字内容\n',
            'generated_ids = [\n',
            '    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)\n',
            ']\n',
            '\n',
            'response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]\n',
            'print(response)'
        ]
    }
]

# Insert before '## 第一阶段通关小结' which is the last text cell (now cell index -1)
data['cells'] = data['cells'][:-1] + new_cells + [data['cells'][-1]]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
