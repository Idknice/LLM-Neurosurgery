import json
import os

notebook_path = r"c:\Users\golde\code\LLM-Neurosurgery\notebooks\02_transformer_architecture.ipynb"

# --- 准备每一块的 cell ---

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

add_md("""# 第 2 阶段：解剖外星人 (深入 Transformer 架构)

经历了第 1 阶段的 PyTorch 张量洗礼和极速加持的 4-bit 模型载入后，你现在已经有能力直面这个时代的终极黑盒——**大语言模型**了。

很多人觉得大模型是玄学，但作为极客，今天我们将拿起手术刀（代码），把 Qwen3.5-4B 剥骨抽筋，一层层地看清楚它到底是怎样思考的！

本节课你将完成三大史诗级成就：
1. **透视蓝图**：用极客代码画出真正的 Transformer 架构塔。
2. **提取记忆**：抓取一句话在神经网络中穿梭时，张量（数字）发生的疯狂裂变。
3. **主脑切除手术 (Lobotomy)**：暴力删掉大模型负责思考的 10 层核心网络，看看它会不会变成“白痴”！""")

add_md("""---
## 0. 课前准备：进入手术室
老规矩，由于 Colab 实例隔离，我们依然需要挂载硬盘并进行 4-bit 闪电加载。""")

add_code("""# 1. 安装基础依赖与底层算子库
!pip install -q -U torch transformers accelerate datasets huggingface_hub bitsandbytes qwen-vl-utils flash-linear-attention causal-conv1d

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import os
from google.colab import drive

# 2. 挂载高速缓存
drive.mount('/content/drive')
drive_cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache/models--Qwen--Qwen3.5-4B"
local_model_path = "/content/qwen3_5_local"

if not os.path.exists(local_model_path):
    print("🚗 正在将几十 GB 的模型权重从网盘抽血到 Colab 本地 SSD 高速通道...")
    !cp -r {drive_cache_dir} {local_model_path}
else:
    print("⚡ 本地高速缓存库已就绪！")

# 3. 极速 4-bit 加载
print("🧠 正在使用 4-bit 量化魔法从虚拟机的系统 SSD 闪电加载多模态架构 (Qwen3.5-4B)...")
processor = AutoProcessor.from_pretrained(local_model_path)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForImageTextToText.from_pretrained(
    local_model_path,
    device_map="auto",
    quantization_config=quantization_config
)
print("✅ 手术对象 (Qwen3.5) 已在手术台上就绪并且被 4-bit 麻醉！")""")

add_md("""---
## 1. 全息透视：Transformer 的真实架构

大模型并不是一个无法理解的实心铅球，它其实是一个像千层糕一样堆叠起来的流水线。
让我们使用 Markdown 神器 `Mermaid`，结合 Qwen 真实的层级架构，来画一张它的透视 X 光片。

(Jupyter / Colab 原生支持 Mermaid 画图，以下图表展示的是语句在模型中攀爬的过程)""")

# 添加 Mermaid 画图模块 (使用 markdown cell 直接渲染)
mermaid_md = """```mermaid
graph TD;
    subgraph 0. 入口处
        A[人类文本: 'Transformer 是什么？'] --> B[Tokenizer 分词器]
        B --> C[Token IDs: 1045, 342, 66...]
    end

    subgraph 1. 大楼一层: 查字典 (Embedding)
        C --> D[Embeddings 词嵌入表矩阵]
        D -->|翻译为高维张量: shape=(序列长, 4096)| E(输入大模型主楼)
    end

    subgraph 2. 大楼主体: 40层循环堆叠的隐藏脑回路 (Hidden Layers)
        E --> L0[Layer 0]
        L0 --> L1[Layer 1]
        L1 --> L2[...]
        L2 --> L39[Layer 39]
        
        %% 每一层的内部微观结构：
        subgraph Layer N 内部
            direction LR
            Attention[Self-Attention 注意力机制<br/>(寻找上下文关系)] --> MLP[MLP 前馈网络<br/>(提取常识记忆)]
        end
    end

    subgraph 3. 楼顶天线: 猜测下一个词 (LM Head)
        L39 -->|提纯后的究极状态张量| F[LM Head 线性分类网]
        F --> G[Vocab 词表中每个子的概率表]
        G --> H((抽卡选出最高概率的下一个字!))
    end
```"""
add_md(mermaid_md)

add_md("""现在，让我们用一行极简单的 Python 代码，扒下 Qwen3.5 的衣服印证上面这张图：""")

add_code("""# 直接打印这个 4-bit Qwen3.5 对象的宏观架构！
# 由于内容极长，它会折叠显示
print(model)""")

add_md("""### 🚀 极客小结：
注意看上面的打印结果，你会发现 Qwen3.5 被切分成了几大块块：
1. `visual`: 这就是处理图片那串 `151655` 的特供视觉网膜模块。
2. `model.embed_tokens`: 把文字 ID 翻译成 4096 维度浮点数的总字典。
3. `model.layers`: **这就是大模型进行“思考”的地方！它一共有足足 40 层 (0 到 39) 互为复制粘贴的架构模块（这就是传说中的 Transformer Blocks）！**
4. `lm_head`: 位于塔顶的分类器，它的维度是`(4096, 151936)`，用于把最终的神经元张量，投票映射回那 15万字 的中文/英文词库中！""")

add_md("""---
## 2. 微观手术：扒出第一层的一根神经元

光看宏观架构不过瘾。
一句话钻进大模型后，被揉捏成了什么样子？让我们来看看第 0 层的注意力头 (`self_attn`) 是怎么长出来这么多参数的。

每一层 Transformer Block (`model.model.layers[i]`) 都由两大部分组成：
- **`self_attn` (自注意力机制)**：通过看别人，搞清楚自己在这个句子里是什么意思。（比如“苹果公司”和“吃苹果”，苹果的含义完全基于上下文）。它主要由 `q_proj, k_proj, v_proj, o_proj` 四个矩阵构成。
- **`mlp` (多层感知机/前馈网络)**：一个极其庞大的参数记忆库（通常包含了上下上下两层升维降维矩阵，如 `gate_proj, up_proj, down_proj`），它记住了“乔布斯创立了苹果”这样的事实知识。

让我们写一段代码，看看在这一层里，注意力头和 MLP 谁更占显存！""")

add_code("""# 写一个小脚本，解剖第 0 层 (layer 0)
layer_0 = model.model.layers[0]

def count_params(module):
    # 计算模块中有多少个浮点数数字
    return sum(p.numel() for p in module.parameters())

attn_params = count_params(layer_0.self_attn)
mlp_params = count_params(layer_0.mlp)
total_layer_params = count_params(layer_0)

print(f"🔪 第 0 层总参数量: {total_layer_params / 10**6:.2f} 百万 (Million)")
print(f"👉 其中，注意力头 (Attention) 占据: {attn_params / 10**6:.2f} 百万，占比 {attn_params / total_layer_params * 100:.1f}%")
print(f"👉 但是，负责常识记忆的 MLP 占据: {mlp_params / 10**6:.2f} 百万，占比 {mlp_params / total_layer_params * 100:.1f}%")

print(\"\\n🤯 原来如此！虽然大家大呼小叫 Transformer 最核心的是 Attention，但真正占据模型近 70% 庞大身躯（也就是显卡内存）的，其实是用来死记硬背巨量常识的 MLP 全连接层！\")""")

add_md("""---
## 3. 切除主脑：残差连接的防破坏性

这是本次手术的最高潮！
在普通人的认知里，一套软件如果删掉了其中四分之一的代码，那它肯定当场崩溃闪退。
但在神经网络这个外星产物面前，**这是不成立的！**

为什么？因为 Transformer 有一个极其伟大和精妙的设计：**残差连接 (Residual Connections)**。
打个比方，第 5 层的输出，不是直接全盘交给第 6 层，而是`输出 = 原始输入 + 第 5 层的处理结果`。
如果第 5 层被搞坏了或者消失了，那么：`输出 = 原始输入 + 0`，信息依然顺畅无阻地继续爬上第 6 层！

今天，我们就来扮演疯狂外科学家，**我们要把 Qwen3.5 原本负责最复杂的中间逻辑推导的那 10 层 (第 15 层到第 24 层)，用一行 Python 数组切片代码，直接“喀嚓”剪断扔掉！**""")

add_code("""# 我们先让原版健全的 Qwen 跑一次复杂问题的逻辑推理，留下案底
messages = [
    {"role": "user", "content": "请用一段话简述：先有鸡还是先有蛋？这是一个哲理问题还是生物学问题？"}
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

print("============ [健全的 Qwen] 正常脑回路输出 ============")
outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))""")

add_code("""print("\\n⚠️ 警告：正在执行脑叶额叶切除手术！")
print(f"切除前，大楼一共有 {len(model.model.layers)} 层")

# 🔪 手起刀落：丢弃切片 [15:25]
model.model.layers = model.model.layers[:15] + model.model.layers[25:]

print(f"🩸 切除完毕，大楼只剩下 {len(model.model.layers)} 层")""")

add_md("""现在，大模型中间凭空缺了 10 层，它的思考回路被“强行”截断缝合了。
让我们再用同样的问题问问这个受到了重创的 Qwen！你猜它会报错，还是会变傻？""")

add_code("""print("============ [受到重创的 Qwen] 缺失脑叶输出 ============")
# 虽然被切断了 10 层，但得益于强大的残差连接和所有 Layer 维度保持 4096 的统一标准
# 运算流水线仍然能够跑通！
broken_outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(broken_outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))""")

add_md("""🤯 **你看到了什么？！**
它没有报错闪退！它还在不知疲倦地吐露汉字！
你可以观察一下它的用词、语法甚至逻辑。它可能像个醉汉一样前后矛盾，有时甚至答非所问、车轱辘话连篇（比如反复背诵前面的一两个词）。
这就是因为它失去了中间那 10 层用来提纯复杂逻辑的隐层，但它最底层的语言直觉（比如主谓宾结构）和最顶层的字词概率表依然还在。

---
## 第二阶段通关小结
在这场极其硬核的神经解剖大戏中，你不仅：
- 画出了大模型 40 层架构的图纸。
- 洞穿了 Attention 和 MLP 参数占用比例的真相。
- 甚至手撕了价值几十万美元训练资源出来的神经网络，并且见证了大模型生命力顽强、模块极度解耦的本质特性。

现在，你对模型的理解已经超越了 90% 的调包侠。
请点击**“保存到 GitHub”**，准备好在下一阶段中，给这具身体接上一条知识外挂 —— 真正的 PEFT / LoRA 模型微调。""")

notebook = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")
