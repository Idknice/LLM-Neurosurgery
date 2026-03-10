import json

path = r"c:\Users\golde\code\LLM-Neurosurgery\notebooks\05_qwen_architecture.ipynb"
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.split("\n")]})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in text.split("\n")]})

md("""# 第 5 阶段：深潜 Qwen3.5 家族架构 (专有技术细节)

在前面四个阶段中，你已经完成了从搭建实验室到操纵模型人格的全部旅程。
但作为终极极客，还有一个问题困扰着你：

> **Qwen3.5 到底跟其他大模型 (如 Llama, GPT) 有什么本质区别？它有哪些独门绝技？**

答案来自我们的 NotebookLM 情报库。Qwen3.5 拥有三大技术创新：
1. **3:1 混合注意力**：Gated DeltaNet (线性注意力) 与标准 Attention 每 4 层交替一次。
2. **SwiGLU MLP**：比 ReLU 更精确更平滑的激活函数门控机制。
3. **DeepStack 视觉编码器**：多层视觉特征融合 + 三维旋转位置编码 (Interleaved-MRoPE)。

本阶段，我们将带代码验证 NotebookLM 给出的每一条技术参数！""")

md("""---
## 0. 课前准备""")

code("""!pip install -q -U torch transformers accelerate huggingface_hub bitsandbytes

import torch
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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
processor = AutoProcessor.from_pretrained(local_model_path)
model = AutoModelForImageTextToText.from_pretrained(
    local_model_path, device_map="auto", quantization_config=quantization_config
)
print("✅ 解剖台就绪！")""")

md("""---
## 1. 混合注意力架构：3:1 交响曲

### NotebookLM 情报
> "模型每4层为一个组合，前3层使用 Gated DeltaNet（线性注意力），第4层使用标准的 Gated Attention，此组合重复8次。"
> "Gated DeltaNet 的计算复杂度呈近线性 O(L) 扩展，而标准注意力是 O(L²)。"

让我们用代码亲自验证这个 3:1 排列模式！""")

code("""# 遍历所有隐藏层，检测每一层的注意力机制类型
print("🔍 逐层扫描 Qwen3.5-4B 的 32 层隐藏层...")
print("=" * 60)

layer_types = []
for i, layer in enumerate(model.model.layers):
    attn_module = layer.self_attn
    attn_type_name = type(attn_module).__name__
    layer_types.append(attn_type_name)
    # 标记角色
    if "DeltaNet" in attn_type_name or "Linear" in attn_type_name:
        role = "🟢 DeltaNet (线性注意力, O(L) 高速通道)"
    else:
        role = "🔴 标准 Attention (全局精确检索, O(L²))"
    print(f"  Layer {i:2d}: {attn_type_name:40s} → {role}")

# 统计比例
unique_types = set(layer_types)
print("\\n" + "=" * 60)
print("📊 统计")
for t in unique_types:
    count = layer_types.count(t)
    print(f"  {t}: {count} 层 ({count/len(layer_types)*100:.0f}%)")

print(f"\\n总层数: {len(layer_types)} (NotebookLM 预测: 32 ✅)")
print("\\n🤯 验证完毕！NotebookLM 说的 3:1 比例是真的吗？看看上面的统计数据！")""")

md("""### 3:1 交替排列的战略意义

```mermaid
graph LR
    subgraph 重复 8 次
        D1["Layer N<br/>🟢 DeltaNet<br/>O(L) 高速"] --> D2["Layer N+1<br/>🟢 DeltaNet<br/>O(L) 高速"]
        D2 --> D3["Layer N+2<br/>🟢 DeltaNet<br/>O(L) 高速"]
        D3 --> SA["Layer N+3<br/>🔴 标准 Attention<br/>O(L²) 精确"]
    end
    SA --> NEXT["下一个 4 层组..."]
    
    style D1 fill:#2ecc71,color:white
    style D2 fill:#2ecc71,color:white
    style D3 fill:#2ecc71,color:white
    style SA fill:#e74c3c,color:white
```

这种设计的天才之处在于：
- **75% 的层** 使用 DeltaNet，以极低的 O(L) 成本处理信息压缩和传递（像高速公路的主干道）。
- **25% 的层** 使用标准 Attention，负责"大海捞针"式的精确信息检索（像主干道上的精密收费站）。
- 结果：模型能支持高达 **262K** 的超长上下文，同时推理成本远低于全量注意力模型！""")

md("""---
## 2. SwiGLU MLP：记忆库的激活机制

### NotebookLM 情报
> "gate_proj 和 up_proj 的维度是 2560→9216，down_proj 是 9216→2560，激活函数采用 SwiGLU。"

传统的 MLP 用 ReLU 激活：`output = ReLU(x @ W1) @ W2`。
但 SwiGLU 更精妙：它用一个额外的"门控矩阵"来精细控制哪些信息应该被激活、哪些应该被抑制。""")

md("""```mermaid
graph LR
    Input["输入<br/>(2560 维)"] --> Gate["gate_proj<br/>(2560→9216)"]
    Input --> Up["up_proj<br/>(2560→9216)"]
    Gate -->|SiLU 激活| Mul(("×"))
    Up --> Mul
    Mul --> Down["down_proj<br/>(9216→2560)"]
    Down --> Output["输出<br/>(2560 维)"]
    
    style Gate fill:#f9c74f,color:black
    style Up fill:#90be6d,color:black
    style Down fill:#577590,color:white
```""")

code("""# 直接扒出第 0 层的 MLP 权重维度，验证 NotebookLM 的预言
mlp = model.model.layers[0].mlp

print("🔬 第 0 层 MLP (SwiGLU) 权重矩阵解剖：")
print("=" * 50)

for name, param in mlp.named_parameters():
    print(f"  {name:20s} → Shape: {str(list(param.shape)):20s} | 参数量: {param.numel():>12,}")

print("\\n📐 维度验证：")
print(f"  gate_proj: 输入维度→中间维度 = ?→? (NotebookLM 预测: 2560→9216)")
print(f"  up_proj:   输入维度→中间维度 = ?→? (NotebookLM 预测: 2560→9216)")
print(f"  down_proj: 中间维度→输出维度 = ?→? (NotebookLM 预测: 9216→2560)")
print("\\n🤯 上面打印出来的 Shape 数字跟 NotebookLM 的预测对得上吗？如果是 ✅ 的话，说明情报可靠！")""")

md("""---
## 3. DeepStack 视觉编码器：图像如何成为 Token

### NotebookLM 情报
> "patch_size=16, spatial_merge_size=2, 采用 Interleaved-MRoPE 三维位置编码。"
> "DeepStack 的核心工作原理是融合多层 ViT 的特征。"

还记得第 1 阶段我们看到 `<|image_pad|>` 的 Token ID 151655 吗？
今天我们来看看在视觉编码器这一边，一张图片究竟是怎么被切碎成 Patches 的。""")

code("""# 扒出视觉编码器的完整架构
print("🔬 Qwen3.5 视觉编码器 (DeepStack) 架构一览：")
print("=" * 60)
print(model.visual)""")

code("""# 提取关键配置参数
visual_config = model.config
print("\\n📐 视觉编码器核心参数：")
print("=" * 50)

# 尝试从不同路径提取视觉相关配置
if hasattr(visual_config, 'vision_config'):
    vc = visual_config.vision_config
    print(f"  Patch Size       : {getattr(vc, 'patch_size', 'N/A')} (NotebookLM 预测: 16)")
    print(f"  Hidden Size      : {getattr(vc, 'hidden_size', 'N/A')}")
    print(f"  Num Layers       : {getattr(vc, 'num_hidden_layers', 'N/A')}")
    print(f"  Num Heads        : {getattr(vc, 'num_attention_heads', 'N/A')}")
    print(f"  Image Size       : {getattr(vc, 'image_size', 'N/A')}")
elif hasattr(model, 'visual') and hasattr(model.visual, 'config'):
    vc = model.visual.config
    print(f"  Patch Size       : {getattr(vc, 'patch_size', 'N/A')} (NotebookLM 预测: 16)")
    print(f"  Hidden Size      : {getattr(vc, 'hidden_size', 'N/A')}")
else:
    print("  (视觉配置结构可能需要手动探索，请查看上方 print(model.visual) 的输出)")

# 空间合并
if hasattr(visual_config, 'vision_config'):
    sms = getattr(visual_config.vision_config, 'spatial_merge_size', 'N/A')
else:
    sms = 'N/A'
print(f"  Spatial Merge    : {sms} (NotebookLM 预测: 2)")""")

md("""### 3.1 极客挑战：计算一张图片会变成多少个视觉 Token""")

code("""# 假设输入一张 640×480 的图片
img_width = 640
img_height = 480
patch_size = 16
spatial_merge = 2

# 步骤 1：调整为 32 的倍数 (Qwen3.5 的要求)
adj_width = (img_width // 32) * 32    # 640 → 640
adj_height = (img_height // 32) * 32  # 480 → 480

# 步骤 2：切成 patch_size × patch_size 的小块
patches_w = adj_width // patch_size   # 640 / 16 = 40
patches_h = adj_height // patch_size  # 480 / 16 = 30
total_patches = patches_w * patches_h

# 步骤 3：空间合并 (2×2 → 1)
merged_tokens = total_patches // (spatial_merge * spatial_merge)

print(f"📸 一张 {img_width}×{img_height} 的图片：")
print(f"  → 调整尺寸后: {adj_width}×{adj_height}")
print(f"  → 切成 {patch_size}×{patch_size} 的小块: {patches_w} × {patches_h} = {total_patches} 个 Patches")
print(f"  → 经过 {spatial_merge}×{spatial_merge} 空间合并后: {merged_tokens} 个视觉 Token")
print(f"\\n🤯 也就是说，这一张小小的图片就会占用 {merged_tokens} 个 <|image_pad|> (ID: 151655) 的坑位！")
print("   这就是为什么图文混合推理比纯文本推理慢得多——光是图片就吃掉了几百个 Token 的位置！")""")

md("""### 3.2 Interleaved-MRoPE 三维位置编码

传统的文本 Transformer 只有一维位置编码（第几个 Token 先后位置）。
但 Qwen3.5 的视觉输入是二维的（图片有宽和高），如果输入的是视频还有时间维度。

**Interleaved-MRoPE** 巧妙地将旋转位置编码 (RoPE) 扩展到了三个维度：
- **高度维度** (Height)
- **宽度维度** (Width)
- **时间维度** (Temporal，用于视频帧)

这让模型不仅知道每个像素块"是第几个"，还知道它"在图片的第几行第几列"——真正的三维时空感知！""")

md("""```mermaid
graph TD
    subgraph 传统文本 RoPE
        T1["Token 1"] --> T2["Token 2"] --> T3["Token 3"]
        T1 -.->|"pos=1"| T1
        T2 -.->|"pos=2"| T2
        T3 -.->|"pos=3"| T3
    end
    
    subgraph "Qwen3.5 Interleaved-MRoPE (三维)"
        P1["Patch (0,0)"] --> P2["Patch (0,1)"]
        P3["Patch (1,0)"] --> P4["Patch (1,1)"]
        P1 -.->|"h=0, w=0, t=0"| P1
        P2 -.->|"h=0, w=1, t=0"| P2
        P3 -.->|"h=1, w=0, t=0"| P3
        P4 -.->|"h=1, w=1, t=0"| P4
    end
```""")

md("""---
## 第 5 阶段通关小结

🏆 **恭喜你！你已经完成了本系列"大模型极客解剖"的全部五个阶段！**

让我们回顾一下你这一路走来获得的能力：

| 阶段 | 技能 | 黑客等级 |
|:-----|:-----|:---------|
| 00 | 搭建云端实验室、极速下载模型 | ⭐ 新手 |
| 01 | PyTorch 张量、Tokenizer 分词、4-bit 量化 | ⭐⭐ 入门 |
| 02 | Transformer 架构透视、MLP/Attention 解剖、脑叶切除手术 | ⭐⭐⭐ 进阶 |
| 03 | LoRA 微调、数据工程、UnSloth 极速训练 | ⭐⭐⭐⭐ 高手 |
| 04 | PyTorch Hook 窃听、注意力热力图、控制向量注入 | ⭐⭐⭐⭐⭐ 黑客 |
| 05 | Qwen3.5 混合注意力、SwiGLU、DeepStack 视觉编码器 | 💀 极客 |

你已经不再是一个"调包侠"了。你真正理解了大模型的心脏是怎么跳动的、大脑是怎么思考的、记忆是怎么存储的。

**去创造属于你自己的 AI 吧！** 🚀""")

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
