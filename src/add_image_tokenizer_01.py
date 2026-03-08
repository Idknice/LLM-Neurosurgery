import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\01_pytorch_huggingface_basics.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find the location to insert the new cells: After the encoded_ids text decoding cell
target_index = -1
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code' and 'encoded_ids = tokenizer.encode(text)' in "".join(cell['source']):
        target_index = i
        break

if target_index != -1:
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.1 极客进阶：图像是如何变成 Token 的？\n",
                "\n",
                "既然我们换上了高级的 `Processor`，你一定会好奇：**文字被分词器变成了 ID 数据，那图片呢？**\n",
                "让我们用前一关问糖果动物的代码，来一探究竟大模型眼中的“图片”到底长什么样！"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import urllib.request\n",
                "from PIL import Image\n",
                "\n",
                "# 我们先把图片下载到内存里作为一个可以直接观察的独立对象\n",
                "img_url = \"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG\"\n",
                "image = Image.open(urllib.request.urlopen(img_url))\n",
                "\n",
                "messages = [\n",
                "    {\n",
                "        \"role\": \"user\",\n",
                "        \"content\": [\n",
                "            {\"type\": \"image\", \"image\": image},  # 注意：这里我们改成了直接喂入 PIL Image 对象\n",
                "            {\"type\": \"text\", \"text\": \"What animal is on the candy? / 这颗糖果上是什么动物？\"}\n",
                "        ]\n",
                "    }\n",
                "]\n",
                "\n",
                "# 把图文混排丢给 Processor 这个黑盒处理\n",
                "print(\"正在启动 Processor 并行处理图像和文字...\")\n",
                "inputs = processor.apply_chat_template(\n",
                "    messages,\n",
                "    add_generation_prompt=True,\n",
                "    return_dict=True,\n",
                "    return_tensors=\"pt\"\n",
                ")\n",
                "\n",
                "print(\"\\n✅ 处理完成！让我们看看 Processor 给大模型准备了什么大餐：\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 重点观察 input_ids：这是包含图文联合编码的一维数组\n",
                "input_ids = inputs[\"input_ids\"][0]\n",
                "print(f\"混合流总长度: {len(input_ids)} 个 Token\\n\")\n",
                "\n",
                "print(\"---- 这个长链里到底混进了什么东西？ ----\")\n",
                "# 我们把这个包含图文 ID 的数组再反向 decode 解码回人类能看懂的形式：\n",
                "decoded_text = processor.decode(input_ids)\n",
                "print(decoded_text)\n",
                "\n",
                "print(\"\\n🤯 发现了吗？！\")\n",
                "print(\"在 Qwen-VL (视觉语言模型) 的眼里，一张图片并不是立体的像素矩阵，\")\n",
                "print(\"而是在文本聊天记录里，硬生生地塞入了一长串的特殊占位符 <|image_pad|>！\")\n",
                "print(\"在这个例子里，Processor 根据这颗糖果图片的尺寸，足足放入了几千个 <|image_pad|> 特殊标记。\")\n",
                "print(\"真正的图像信息，会附带在另外一个叫做 inputs['pixel_values'] 的立体张量中，\")\n",
                "print(\"大模型在阅读那些 <|image_pad|> 时，会将目光瞬间转向 pixel_values 里的图像块！\")"
            ]
        }
    ]
    
    # Insert new cells
    data['cells'] = data['cells'][:target_index+1] + new_cells + data['cells'][target_index+1:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
else:
    print("Could not find insertion point")
