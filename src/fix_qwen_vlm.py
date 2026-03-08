import json
import os

project_root = r"c:\Users\golde\code\LLM-Neurosurgery"

def update_00_notebook():
    filepath = os.path.join(project_root, 'notebooks', '00_cloud_lab_setup.ipynb')
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Cell 6: update pip install to include git version of transformers just in case Qwen3.5 needs edge version
    for cell in data['cells']:
        if cell['cell_type'] == 'code' and '!pip install' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                if '!pip install -q -U torch transformers' in line:
                    # Also install qwen-vl-utils and newest transformers if necessary
                    line = "!pip install -q -U torch transformers accelerate datasets huggingface_hub qwen-vl-utils git+https://github.com/huggingface/transformers.git\n"
                new_source.append(line)
            cell['source'] = new_source

    # Replace model loading cell
    for cell in data['cells']:
        source_str = ''.join(cell['source'])
        if 'AutoModelForCausalLM' in source_str:
            new_source = [
                'from transformers import AutoProcessor, AutoModelForImageTextToText\n',
                'import torch\n',
                '\n',
                'print("🧠 正在将模型加载到 GPU 显存中，此时如果开启一个新终端运行 nvidia-smi 能够看到显存占用飙升...")\n',
                '\n',
                'cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache"\n',
                'model_id = "Qwen/Qwen3.5-4B"\n',
                '\n',
                '# 对于多模态模型，我们使用 Processor 替代纯文本的 Tokenizer\n',
                'processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)\n',
                '\n',
                '# 使用 AutoModelForImageTextToText 加载多模态基座\n',
                'model = AutoModelForImageTextToText.from_pretrained(\n',
                '    model_id,\n',
                '    cache_dir=cache_dir,\n',
                '    torch_dtype="auto",\n',
                '    device_map="auto"\n',
                ')\n',
                '\n',
                'print("✅ 模型加载成功！显存已经全部就绪！")'
            ]
            cell['source'] = new_source

    # Replace chat generation cell
    for cell in data['cells']:
        source_str = ''.join(cell['source'])
        if 'messages = [' in source_str and 'apply_chat_template' in source_str:
            new_source = [
                'import time\n',
                '\n',
                '# 多模态模型的输入结构：支持图片+文本混合\n',
                'messages = [\n',
                '    {\n',
                '        "role": "user",\n',
                '        "content": [\n',
                '            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},\n',
                '            {"type": "text", "text": "What animal is on the candy? / 这颗糖果上是什么动物？"}\n',
                '        ]\n',
                '    }\n',
                ']\n',
                '\n',
                '# 使用 Processor 将多模态消息打包\n',
                'inputs = processor.apply_chat_template(\n',
                '    messages,\n',
                '    add_generation_prompt=True,\n',
                '    tokenize=True,\n',
                '    return_dict=True,\n',
                '    return_tensors="pt"\n',
                ').to(model.device)\n',
                '\n',
                'print("\\n================ 🤖 Qwen3.5 正在 GPU 上凝视图片并吟唱 =================\\n")\n',
                '\n',
                'start_time = time.time()\n',
                '\n',
                '# 生成回答\n',
                'outputs = model.generate(\n',
                '    **inputs,\n',
                '    max_new_tokens=40\n',
                ')\n',
                '\n',
                'end_time = time.time()\n',
                '\n',
                '# 解码还原输出（利用 inputs["input_ids"].shape[-1]: 优雅剥离前面输入的问题部分）\n',
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

def update_01_notebook():
    filepath = os.path.join(project_root, 'notebooks', '01_pytorch_huggingface_basics.ipynb')
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Update markdown section strings about tokenizer to processor
    for cell in data['cells']:
        if cell['cell_type'] == 'markdown':
            source_str = ''.join(cell['source'])
            if '兵器二：Tokenizer' in source_str:
                cell['source'] = [s.replace('Tokenizer (分词器)', 'Processor (处理器与分词器)') for s in cell['source']]
    
    # Update tokenizer load cell to AutoProcessor
    for cell in data['cells']:
        source_str = ''.join(cell['source'])
        if 'AutoTokenizer' in source_str and 'from_pretrained' in source_str:
            new_source = [
                'from transformers import AutoProcessor\n',
                '\n',
                'model_id = "Qwen/Qwen3.5-4B"\n',
                'cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache"\n',
                '\n',
                'print("正在加载 Processor (整合了分词器和图像预处理器)...")\n',
                'processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)\n',
                '\n',
                '# 获取底层的文本分词器看看\n',
                'tokenizer = processor.tokenizer\n',
                'print(f"Qwen3.5 词表大小: {tokenizer.vocab_size} 个词块")'
            ]
            cell['source'] = new_source

    # Update quantization model load cell
    for cell in data['cells']:
        source_str = ''.join(cell['source'])
        if 'AutoModelForCausalLM' in source_str and 'BitsAndBytesConfig' in source_str:
            new_source = [
                'from transformers import AutoModelForImageTextToText, BitsAndBytesConfig\n',
                'import torch\n',
                '\n',
                'print("正在使用 4-bit 量化魔法加载多模态架构 (Qwen3.5-4B)...")\n',
                'quantization_config = BitsAndBytesConfig(\n',
                '    load_in_4bit=True,\n',
                '    bnb_4bit_compute_dtype=torch.bfloat16,\n',
                '    bnb_4bit_quant_type="nf4"\n',
                ')\n',
                '\n',
                '# 注意：这是多模态模型的专属类名\n',
                'model_4bit = AutoModelForImageTextToText.from_pretrained(\n',
                '    model_id,\n',
                '    cache_dir=cache_dir,\n',
                '    device_map="auto",\n',
                '    quantization_config=quantization_config\n',
                ')\n',
                '\n',
                'print("\\n✅ 量化加载完成！")'
            ]
            cell['source'] = new_source

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")

if __name__ == '__main__':
    update_00_notebook()
    update_01_notebook()
    print("Notebooks updated successfully.")
