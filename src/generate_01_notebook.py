import json
import os

def create_notebook(filepath):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 第 1 阶段：兵器库储备 (PyTorch 与 Hugging Face 基础)\n",
                    "\n",
                    "大语言模型 (LLM) 看似神秘莫测，但剖开华丽的表象，其底层不过是极其庞大但规则简单的**矩阵乘法**。而在人工智能领域，处理矩阵的终极兵器就是 **PyTorch**。\n",
                    "\n",
                    "在真正解剖 Transformer 之前，我们必须熟练掌握两把基本的手术刀：\n",
                    "1. **PyTorch 张量 (Tensor)**：大模型的血液和骨骼。\n",
                    "2. **Tokenizer (分词器)**：人类语言与矩阵数字交互的翻译官。\n",
                    "3. **量化魔法 (Quantization)**：把庞然大物压缩进平民显卡的黑科技。\n",
                    "\n",
                    "---\n",
                    "## 兵器一：PyTorch 与显存转移"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1.1 万物皆张量 (Tensor)\n",
                    "在 PyTorch 中，所有的数据——无论是文本、图片还是模型的参数——全都被定义为张量 (Tensor)。它和 Numpy 的多维数组极为相似，但它拥有一个超级能力：**可以在 GPU 上进行闪电般的并行计算**。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "\n",
                    "# 创建一个简单的 2x3 张量（你可以把它看作二维矩阵）\n",
                    "tensor_a = torch.tensor([\n",
                    "    [1.0, 2.0, 3.0], \n",
                    "    [4.0, 5.0, 6.0]\n",
                    "])\n",
                    "\n",
                    "print(\"张量数据:\\n\", tensor_a)\n",
                    "print(\"\\n张量维度 (Shape):\", tensor_a.shape)   # 极度重要！以后 debug 70% 的时间在对 shape\n",
                    "print(\"数据类型 (Dtype):\", tensor_a.dtype)   # FP32 (float32) 是默认精度\n",
                    "print(\"所在设备 (Device):\", tensor_a.device) # 默认创建在内存 (CPU) 中"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1.2 设备转移：显存报错的根源\n",
                    "初学者最常碰到的报错叫 `RuntimeError: Expected all tensors to be on the same device`。这是因为**在内存中的张量和在显存中的张量是不能直接相乘的**。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 把张量发射到 GPU 上\n",
                    "if torch.cuda.is_available():\n",
                    "    tensor_gpu = tensor_a.to(\"cuda\")\n",
                    "    print(\"转移后所在设备:\", tensor_gpu.device)\n",
                    "else:\n",
                    "    print(\"没有检测到 GPU，你是不是忘了在 Colab 开启 T4 硬件加速？\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1.3 极客加餐：反向传播 (Autograd) 体验\n",
                    "大模型为什么具有“学习能力”？一切魔法的根源是微积分的**链式求导法则**。PyTorch 提供了一个叫做 Autograd 的作弊器，它可以自动计算偏导数。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 我们创建一个需要计算梯度的张量（模拟模型里的一根神经元）\n",
                    "x = torch.tensor([2.0], requires_grad=True)\n",
                    "\n",
                    "# 定义一个非常简单的数学运算 y = x^2 + 3x\n",
                    "y = x**2 + 3*x\n",
                    "\n",
                    "# 根据微积分，dy/dx = 2x + 3。当 x=2 时，梯度应该是 7\n",
                    "# 让 PyTorch 自动帮我们反向求导\n",
                    "y.backward()\n",
                    "\n",
                    "print(f\"PyTorch 算出的 x 的梯度 (导数): {x.grad.item()}\")\n",
                    "print(\"🤯 以后所有的模型微调 (Fine-tuning)，本质上就是海量的神经元在做上面这几行代码！\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 兵器二：Tokenizer (分词器)\n",
                    "\n",
                    "大语言模型**根本不认识中文和英文**，它是一台纯血的计算器，只认数字。Tokenizer 的作用就是把我们的长句子“剁碎” (Tokenize) 并翻译成大模型字典里的 ID 数字。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from transformers import AutoTokenizer\n",
                    "\n",
                    "model_id = \"Qwen/Qwen3.5-4B\"\n",
                    "cache_dir = \"/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache\"\n",
                    "\n",
                    "print(\"正在加载分词器...\")\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)\n",
                    "\n",
                    "# 看看 Qwen 的脑子里装了多少个基础词汇 (Vocab)\n",
                    "print(f\"Qwen3.5 词表大小: {tokenizer.vocab_size} 个词块\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "text = \"大模型极客实战，从Colab起飞！ LLM is awesome!\"\n",
                    "\n",
                    "# 编码：人类语言 -> 机器 ID\n",
                    "encoded_ids = tokenizer.encode(text)\n",
                    "print(\"原本的句子:\", text)\n",
                    "print(\"Token IDs:\", encoded_ids)\n",
                    "print(f\"文字长度: {len(text)} 字符，Token 数量: {len(encoded_ids)}\")\n",
                    "\n",
                    "print(\"\\n--- 解剖看看每个 ID 对应什么字 ---\")\n",
                    "for tid in encoded_ids:\n",
                    "    print(f\"{tid} \\t -> \\t '{tokenizer.decode([tid])}'\")\n",
                    "\n",
                    "# 提示：这就是为什么某些英文单词模型怎么也算不对字母个数，因为它脑子里整个单词就是一个数字编码 (例如 awesome)。"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 兵器三：量化魔法 (Quantization)\n",
                    "\n",
                    "在这个课程里我们要解剖的是 4B 模型 (40亿参数)。\n",
                    "\n",
                    "如果我们用默认的纯正高精度浮点数 (FP32，每个数字占 4 个字节) 来加载它，它需要 `4,000,000,000 * 4 Bytes ≈ 16 GB` 的显存。再加上推理时的计算损耗，**免费的 15GB T4 显卡当场就会报 OOM (Out of Memory) 显存溢出错误而崩溃**。\n",
                    "\n",
                    "救星就是 **`bitsandbytes`** 库，它能把高精度浮点数压缩成 4 位 (4-bit, 占 0.5个字节) 整数，让模型体积缩小近 8 倍，同时智力几乎不打折扣！"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 我们先看一下清爽状态下的显卡显存占用情况\n",
                    "!nvidia-smi | grep MiB"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n",
                    "import torch\n",
                    "\n",
                    "print(\"正在使用 4-bit 量化魔法加载 Qwen3.5-4B...\")\n",
                    "# 配置量化参数: 使用极端的 4bit NF4 数据类型加载\n",
                    "quantization_config = BitsAndBytesConfig(\n",
                    "    load_in_4bit=True,\n",
                    "    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时用 16 位精度保证性能\n",
                    "    bnb_4bit_quant_type=\"nf4\"\n",
                    ")\n",
                    "\n",
                    "model_4bit = AutoModelForCausalLM.from_pretrained(\n",
                    "    model_id,\n",
                    "    cache_dir=cache_dir,\n",
                    "    device_map=\"auto\",\n",
                    "    quantization_config=quantization_config\n",
                    ")\n",
                    "\n",
                    "print(\"\\n✅ 量化加载完成！\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def print_trainable_parameters(model):\n",
                    "    \"\"\"\n",
                    "    打印模型参数的数据\n",
                    "    \"\"\"\n",
                    "    trainable_params = 0\n",
                    "    all_param = 0\n",
                    "    for _, param in model.named_parameters():\n",
                    "        all_param += param.numel()\n",
                    "        if param.requires_grad:\n",
                    "            trainable_params += param.numel()\n",
                    "    print(\n",
                    "        f\"全模型参数量: {all_param:,} | 可训练参数量: {trainable_params:,} | \"\n",
                    "        f\"占比: {100 * trainable_params / all_param:.4f}%\"\n",
                    "    )\n",
                    "\n",
                    "print_trainable_parameters(model_4bit)\n",
                    "print(\"你看，因为被 4bit 压缩了，当前模型的所有层都是被冻结（不可计算梯度修饰）的，可训练参数占 0%\")\n",
                    "print(\"\\n最后，我们再看一下现在的显存，是不是只有极其可怜的几个 G 的占用？\")\n",
                    "!nvidia-smi | grep MiB"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 第 1 阶段通关小结\n",
                    "你现在已经拥有了一本厚重的《解剖学图谱》：明白了一切皆是 `Tensor`，懂得了语言是被 `Tokenizer` 切碎的数字，并且学会了像黑客一样用 `4-bit 量化` 把庞然大物硬塞进贫民窟的显卡中。\n",
                    "\n",
                    "下一阶段（第 2 阶段），我们将亲手把这只已经被压缩好的 Qwen 剥骨抽筋，找出它进行“逻辑思考”的大脑核心（Attention 和 MLP），甚至我们将暴力切掉它的一个脑叶，看看它会变成什么样！🔥"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Notebook created at {filepath}")

if __name__ == '__main__':
    project_root = r"c:\Users\golde\code\LLM-Neurosurgery"
    notebooks_dir = os.path.join(project_root, "notebooks")
    filepath = os.path.join(notebooks_dir, "01_pytorch_huggingface_basics.ipynb")
    create_notebook(filepath)
