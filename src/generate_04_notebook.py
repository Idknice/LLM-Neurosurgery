import json

path = r"c:\Users\golde\code\LLM-Neurosurgery\notebooks\04_representation_engineering.ipynb"
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.split("\n")]})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in text.split("\n")]})

md("""# 第 4 阶段：前沿极客实验 (机制可解释性与干预)

如果说第 2 阶段是"看清大脑结构图"，第 3 阶段是"往大脑里灌新知识"，那么今天的第 4 阶段就是真正的**赛博朋克黑客操作**——

我们要做三件事：
1. **窃听**：在模型的神经突触上安装"窃听器" (PyTorch Hook)，偷取它在思考时的隐藏信号。
2. **透视**：把注意力头的权重画成热力图，看清模型在阅读一句话时究竟在关注谁。
3. **操纵**：计算并注入一个"控制向量"，像幕后黑手一样篡改模型的输出风格！""")

md("""---
## 0. 课前准备""")

code("""!pip install -q -U torch transformers accelerate huggingface_hub bitsandbytes matplotlib seaborn

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from google.colab import drive
import os

drive.mount('/content/drive')
drive_cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache/models--Qwen--Qwen3.5-4B"
local_model_path = "/content/qwen3_5_local"

if not os.path.exists(local_model_path):
    print("🚗 高速拉取模型到本地 SSD...")
    !cp -r {drive_cache_dir} {local_model_path}
else:
    print("⚡ 本地缓存就绪！")

# 4-bit 加载
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
processor = AutoProcessor.from_pretrained(local_model_path)
model = AutoModelForImageTextToText.from_pretrained(
    local_model_path, device_map="auto", quantization_config=quantization_config
)
print("✅ 手术台就绪！")""")

md("""---
## 1. PyTorch Hook：安装神经窃听器

在 PyTorch 中，`register_forward_hook` 允许你在模型的**任意一层**上挂载一个回调函数。
每当数据流经这一层时，你的回调函数就会被自动触发，并且能拿到该层的输入和输出张量。

这就像在神经网络的电缆上架设了一台窃听设备——模型自己完全不知道你在偷看它的思想！""")

md("""```mermaid
graph LR
    Input["输入张量"] --> Layer["model.layers[15]"]
    Layer --> Output["输出张量"]
    Layer -.->|"🎧 Hook 偷听!"| Hook["你的回调函数<br/>captured_states = output"]
    style Hook fill:#ff6b6b,color:white
```""")

code("""# 我们的"窃听器"存储容器
captured_hidden_states = {}

def make_hook(layer_name):
    \"\"\"工厂函数：为指定层创建一个窃听器\"\"\"
    def hook_fn(module, input, output):
        # output 可能是 tuple，取第一个元素（即 hidden states 张量）
        if isinstance(output, tuple):
            captured_hidden_states[layer_name] = output[0].detach().cpu()
        else:
            captured_hidden_states[layer_name] = output.detach().cpu()
    return hook_fn

# 在第 0 层和第 15 层安装窃听器
hooks = []
hooks.append(model.model.layers[0].register_forward_hook(make_hook("layer_0")))
hooks.append(model.model.layers[15].register_forward_hook(make_hook("layer_15")))

print("🎧 窃听器已在 Layer 0 和 Layer 15 上安装完毕！")""")

code("""# 让模型处理一句话，触发窃听器
test_text = "大语言模型的本质是一台超级概率计算器。"
messages = [{"role": "user", "content": test_text}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

# 只做一次前向传播（不需要生成完整回答），纯粹是为了触发 Hook
with torch.no_grad():
    _ = model(**inputs)

print(f"✅ 窃听成功！捕获了 {len(captured_hidden_states)} 层的隐藏状态")
for name, tensor in captured_hidden_states.items():
    print(f"  📡 {name}: shape = {tensor.shape}, dtype = {tensor.dtype}")""")

md("""### 1.1 语义漂移可视化：同一个词在不同层的"长相"变化

一个 Token 在进入第 0 层时，它只是一个朴素的词嵌入向量。
但当它穿越了 15 层 Transformer Block 后，它已经吸收了海量的上下文信息，变成了一个完全不同的"生物"。

让我们直接比较同一个 Token 在两层之间的余弦相似度！""")

code("""from torch.nn.functional import cosine_similarity

# 取序列中间位置的某个 token
token_idx = inputs["input_ids"].shape[-1] // 2

vec_layer_0 = captured_hidden_states["layer_0"][0, token_idx]   # shape: (hidden_dim,)
vec_layer_15 = captured_hidden_states["layer_15"][0, token_idx]

sim = cosine_similarity(vec_layer_0.unsqueeze(0), vec_layer_15.unsqueeze(0))
print(f"Token 位置 #{token_idx} 在 Layer 0 vs Layer 15 的余弦相似度: {sim.item():.4f}")
print("\\n🤯 如果相似度远低于 1.0，说明这个 Token 的'含义'在经过 15 层思考后已经发生了巨大的语义漂移！")

# 清除 Hook，释放资源
for h in hooks:
    h.remove()
print("\\n🧹 窃听器已安全拆除。")""")

md("""---
## 2. 注意力热力图：模型在看谁？

在 Transformer 中，Self-Attention 的核心就是"每个词都在观察其他所有词，并决定谁对自己最重要"。

我们可以提取注意力权重矩阵，并将其画成热力图。
如果位置 (i, j) 的颜色很深，说明第 i 个 Token 在"思考"自己是什么意思时，重度参考了第 j 个 Token 的信息。""")

code("""# 重新进行一次推理，这次我们需要的是注意力权重
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# 提取最后一层标准注意力层的注意力权重
# Qwen3.5 的混合架构中，每第 4 层是标准注意力层（3:1 排列）
# 最后一个标准注意力层的索引
attn_weights = None
for i in range(len(outputs.attentions) - 1, -1, -1):
    if outputs.attentions[i] is not None:
        attn_weights = outputs.attentions[i]
        print(f"找到注意力权重来自第 {i} 层")
        break

if attn_weights is not None:
    # 取第一个注意力头的权重
    attn = attn_weights[0, 0].detach().cpu().float().numpy()  # (seq_len, seq_len)

    # 解码 token 文本用于标注坐标轴
    token_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = [processor.decode([tid]) for tid in token_ids]
    # 限制显示前 30 个 token 避免图片太大
    max_display = min(30, len(tokens))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attn[:max_display, :max_display],
        xticklabels=tokens[:max_display],
        yticklabels=tokens[:max_display],
        cmap="YlOrRd",
        square=True
    )
    plt.title("Attention Heatmap (Head 0)", fontsize=14)
    plt.xlabel("Key (被关注的 Token)")
    plt.ylabel("Query (正在思考的 Token)")
    plt.xticks(fontsize=8, rotation=45, ha='right')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()
    print("\\n📊 颜色越深 = 关注度越高。看看模型在解读每个词时，最依赖哪些上下文！")
else:
    print("⚠️ 未能提取到注意力权重，Qwen3.5 的 DeltaNet 层可能不输出标准注意力矩阵。")
    print("这恰好验证了 DeltaNet 线性注意力与标准注意力的本质区别！")""")

md("""---
## 3. 控制向量注入：操纵模型的"人格"

这是本课程中最前沿、最黑客的实验。

**核心思想** (来自 Representation Engineering 论文)：
- 大模型的"情绪"、"风格"、"态度"等抽象特征，其实都编码在中间层的 Hidden States 中。
- 如果我们能提取出"积极回答"和"消极回答"在某一层的 Hidden States 均值差异向量，并在推理时把这个向量注入，就能像调节旋钮一样控制模型的输出倾向。""")

md("""```mermaid
graph TD
    A["正面 Prompt 组 (积极、乐观)"] -->|前向传播提取| P["正面 Hidden States 均值"]
    B["负面 Prompt 组 (消极、悲观)"] -->|前向传播提取| N["负面 Hidden States 均值"]
    P --> Diff["控制向量 = 正面均值 - 负面均值"]
    N --> Diff
    Diff -->|推理时加到中间层| Model["Qwen3.5 Layer 15 输出"]
    Model --> Result["模型被'鬼手'操纵了！"]
    style Diff fill:#f9c74f,color:black
    style Result fill:#ff6b6b,color:white
```""")

code("""# 步骤一：准备正面/负面 prompt 组
positive_prompts = [
    "生活真的很美好，每天都充满了希望和阳光。请谈谈你的感受。",
    "我今天获得了一个很棒的成就，内心非常喜悦！你怎么看？",
    "世界上有太多值得感恩的事情了。说说你认为最美好的事。",
]
negative_prompts = [
    "一切都毫无意义，活着真的很累。请谈谈你的感受。",
    "我今天又失败了，什么都做不好。你怎么看？",
    "这个世界充满了不公和痛苦。说说你认为最糟糕的事。",
]

def extract_hidden_state(prompts, layer_idx=15):
    \"\"\"提取一组 prompt 在指定层的 Hidden States 均值\"\"\"
    all_states = []
    hook_storage = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hook_storage["state"] = output[0].detach().cpu()
        else:
            hook_storage["state"] = output.detach().cpu()

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        inp = processor.apply_chat_template(
            msgs, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            _ = model(**inp)
        # 取最后一个 token 的 hidden state 作为该 prompt 的代表
        all_states.append(hook_storage["state"][0, -1, :])

    hook.remove()
    return torch.stack(all_states).mean(dim=0)  # 返回均值向量

print("🔬 正在提取正面情绪组的隐藏状态...")
pos_mean = extract_hidden_state(positive_prompts)
print("🔬 正在提取负面情绪组的隐藏状态...")
neg_mean = extract_hidden_state(negative_prompts)

# 计算控制向量
steering_vector = pos_mean - neg_mean
print(f"\\n✅ 控制向量已计算完毕！Shape: {steering_vector.shape}")
print(f"控制向量的 L2 范数: {steering_vector.norm().item():.2f}")""")

code("""# 步骤二：在推理时注入控制向量
# 我们用一个中性的问题来测试操纵效果

neutral_question = "你对人生的看法是什么？"

def generate_with_steering(question, vector, scale=3.0, layer_idx=15):
    \"\"\"在指定层注入控制向量后生成回答\"\"\"
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            modified = output[0] + scale * vector.to(output[0].device)
            return (modified,) + output[1:]
        else:
            return output + scale * vector.to(output.device)

    hook = model.model.layers[layer_idx].register_forward_hook(steering_hook)

    msgs = [{"role": "user", "content": question}]
    inp = processor.apply_chat_template(
        msgs, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=150)

    hook.remove()
    return processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)

# 对照组：无操纵
print("============ [正常模式] 无操纵 ============")
msgs = [{"role": "user", "content": neutral_question}]
inp = processor.apply_chat_template(
    msgs, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=150)
print(processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))

# 实验组：注入正面控制向量
print("\\n============ [鬼手操纵] 注入正面情绪控制向量 ============")
print(generate_with_steering(neutral_question, steering_vector, scale=3.0))

# 实验组：注入负面控制向量（反向）
print("\\n============ [鬼手操纵] 注入负面情绪控制向量 ============")
print(generate_with_steering(neutral_question, -steering_vector, scale=3.0))""")

md("""🤯 **看到了吗？！**

同一个中性提问，加上了正面控制向量后模型变得阳光积极，反过来加上负面控制向量后，模型开始流露出消极悲观的倾向。

这就是 **Representation Engineering** 的精髓：大模型的"人格"和"情绪"并不是玄学，而是存储在中间层的特定数字特征中。
只要你知道怎么找到它并拨动它，你就可以像调节旋钮一样控制大模型的输出风格。

---
## 第 4 阶段通关小结

恭喜你，你已经掌握了大模型黑客的三大核心技能：
1. **窃听 (Hook)**：在任意层挂载窃听器，实时抓取神经信号。
2. **透视 (Attention Map)**：画出注意力权重，看穿模型的"视线"分布。
3. **操纵 (Steering Vector)**：计算并注入控制向量，像幕后黑手一样篡改模型的输出！

下一阶段（第 5 阶段），我们将深潜 Qwen3.5 的专属技术细节：它独特的 3:1 混合注意力架构 (Gated DeltaNet)、SwiGLU 激活函数、以及 DeepStack 视觉编码器的图像 Patch 切分机制！ 🔥""")

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
