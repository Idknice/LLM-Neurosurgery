import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\01_pytorch_huggingface_basics.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Locate the cell where the outputs of input_ids are decoded
target_index = -1
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code' and 'input_ids = inputs["input_ids"][0]' in "".join(cell['source']):
        target_index = i
        break

if target_index != -1:
    source = [
        '# 重点观察 input_ids：这是包含图文联合编码的一维数组\n',
        'input_ids = inputs["input_ids"][0]\n',
        'print(f"混合流总长度: {len(input_ids)} 个 Token\\n")\n',
        '\n',
        'print("---- 这个长链里到底混进了什么东西？ ----")\n',
        '# 我们把这个包含图文 ID 的数组反向 decode 解码回人类能看懂的形式：\n',
        'decoded_text = processor.decode(input_ids)\n',
        'print(decoded_text)\n',
        '\n',
        'print("\\n🤯 发现了吗？！")\n',
        'print("在 Qwen-VL (视觉语言模型) 的眼里，一张图片并不是立体的像素矩阵，")\n',
        'print("而是在文本聊天记录里，硬生生地塞入了一长串的特殊占位符 <|image_pad|>！")\n',
        '\n',
        '# 让我们找出那些特殊的 image_pad 的 真身 (Token ID 号码是多少)！\n',
        'print("\\n--- 探究 <|image_pad|> 的底牌 ---")\n',
        '# 在分词器内部查找名为 "<|image_pad|>" 的映射号码\n',
        'image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")\n',
        '\n',
        '# 统计我们刚才那张糖果图片到底占用了多少个这样的 Token 碎片\n',
        'num_image_tokens = (input_ids == image_pad_id).sum().item()\n',
        '\n',
        'print(f"特殊字符串 `<|image_pad|>` 在 Qwen3.5 字典里的真实 ID 是: {image_pad_id}")\n',
        'print(f"我们传入的这张糖果图片，被切分成了 {num_image_tokens} 块，占用了足足 {num_image_tokens} 个连在一起的图像占位 ID！")\n',
        '\n',
        'print("\\n--- 查看原生 Input ID 数组的前 50 个元素 ---")\n',
        'print("你能在这里面看到连续的 image_pad_id (比如 151655) 吗？")\n',
        '# 打印前 50 个 token id 让大家能肉眼看见那一长串相同的数字\n',
        'print(input_ids[:50].tolist())'
    ]
    data['cells'][target_index]['source'] = source
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Notebook updated successfully.")
else:
    print("Could not find the target cell to update.")
