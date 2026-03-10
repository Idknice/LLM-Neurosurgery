import json

path = r"c:\Users\golde\code\LLM-Neurosurgery\notebooks\03_finetuning_lora.ipynb"
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.split("\n")]})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in text.split("\n")]})

md("""# 第 3 阶段：记忆植入与性格重塑 (LoRA 微调实战)

在前两个阶段中，你已经亲手解剖了 Qwen3.5 的大脑架构，见证了 40 层 Transformer Block 的壮观堆叠和残差连接的暴力美学。

现在，问题来了：**那一坨参数里存储的"知识"，是人类在训练阶段硬灌进去的。如果我想让它学会新的知识呢？**

比如，我想让 Qwen 不再是一个泛泛的聊天机器人，而是变成一个能够回答医学专业问答的"AI 主治医师"。

本阶段你将完成：
1. **认知升级**：理解为什么全参微调在免费显卡上完全行不通。
2. **数据工程**：将原始数据集加工成大模型能吃下的"饲料"格式。
3. **LoRA 注入**：使用 UnSloth 框架在 T4 上完成极速 4-bit LoRA 微调。
4. **效果验证**：对比"原始 Qwen" vs "植入记忆后的 Qwen" 的回答差距。""")

md("""---
## 0. 课前准备""")

code("""# 安装 UnSloth（Qwen3.5 官方推荐的免费微调框架）和必需依赖
!pip install -q -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q -U torch transformers accelerate datasets huggingface_hub bitsandbytes trl peft

import torch
from google.colab import drive
import os

drive.mount('/content/drive')
drive_cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache/models--Qwen--Qwen3.5-4B"
local_model_path = "/content/qwen3_5_local"

if not os.path.exists(local_model_path):
    print("🚗 正在将模型权重从网盘高速拉取到本地 SSD...")
    !cp -r {drive_cache_dir} {local_model_path}
else:
    print("⚡ 本地高速缓存已就绪！")

print("✅ 环境准备完毕！")""")

md("""---
## 1. 为什么不能全参训练？一道致命算术题

让我们算一笔账。在第 2 阶段你已经知道 Qwen3.5-4B 有大约 40 亿个参数。

全参数微调(Full Fine-tuning) 时，优化器 (如 AdamW) 需要为**每一个参数**额外保存：
- 1 份梯度 (gradient)
- 1 份一阶动量 (first moment)
- 1 份二阶动量 (second moment)

于是总内存 ≈ `参数量 × (权重 + 梯度 + 2个动量) × 每个数字的字节数`。
用 FP32 (4 bytes)：`4B × 4 × 4 bytes ≈ 64 GB`。

**T4 显卡只有 15GB，当场爆炸 💥**

所以，我们需要一种只改动极少量参数就能让模型学到新知识的方法——""")

md("""### 1.1 LoRA：在大脑旁边贴便签条

**LoRA (Low-Rank Adaptation)** 的核心思想极其优雅：
- 不直接修改原始的巨型权重矩阵 $W$ (比如 4096×4096)
- 而是在旁边"夹带"两个极其扁平的小矩阵 $A$ (4096×16) 和 $B$ (16×4096)
- 推理时，实际权重 = $W + A \\times B$

两个矩阵的秩 (`rank=16`) 远远小于原始矩阵维度，所以需要训练的参数量从数十亿骤降到**几百万**——T4 瞬间就能驾驭！""")

md("""```mermaid
graph LR
    subgraph 原始冻结权重
        W["W (4096×4096)<br/>❄️ 完全冻结不动"]
    end
    subgraph LoRA 便签条
        A["A (4096×r)"] --> B["B (r×4096)"]
    end
    Input --> W
    Input --> A
    W --> Plus((+))
    B --> Plus
    Plus --> Output
```""")

md("""---
## 2. 数据工程：制作大模型的"饲料"

再厉害的模型，喂了垃圾数据也只会产出垃圾。
微调数据集的格式必须是标准的**对话结构 (Chat Format)**，每一条数据都是一段人和AI的对话。""")

code("""from datasets import load_dataset

# 我们使用一个开源的中文医学问答数据集作为演示
# (如果 Hugging Face 连不上，可以替换为任何其他中文问答数据集)
dataset = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1", split="train")

# 看看数据集长什么样
print(f"数据集总量: {len(dataset)} 条")
print("\\n--- 第一条数据预览 ---")
print(dataset[0])""")

code("""# 为了在免费 T4 上快速体验，我们只取前 500 条数据进行微调
dataset = dataset.select(range(500))

# 定义格式化函数：把原始数据转化成 Qwen 标准聊天模板
def format_chat(example):
    \"\"\"将一条数据转换成 Qwen 能消化的聊天格式\"\"\"
    messages = []
    # 添加系统提示（可选但推荐）
    messages.append({"role": "system", "content": "你是一名专业的医学AI助手，请根据医学知识准确回答用户的健康问题。"})
    # 添加用户问题
    messages.append({"role": "user", "content": example.get("instruction", example.get("input", ""))})
    # 添加 AI 回答
    messages.append({"role": "assistant", "content": example.get("output", "")})
    return {"messages": messages}

dataset = dataset.map(format_chat)
print("\\n--- 格式化后的第一条数据 ---")
print(dataset[0]["messages"])""")

md("""---
## 3. LoRA 极速微调：UnSloth 上阵

[UnSloth](https://github.com/unslothai/unsloth) 是 Qwen 官方推荐的免费微调框架，它的核心优势是：
- 自动 4-bit 量化加载
- 比 Hugging Face 原生 `Trainer` 快 2-5 倍
- 内存占用极度压缩，T4 上流畅运行""")

code("""from unsloth import FastLanguageModel

# 使用 UnSloth 极速加载 4-bit 量化版 Qwen3.5
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=2048,       # 最大支持的上下文长度
    dtype=None,                # 自动检测
    load_in_4bit=True,         # 4-bit 量化加载
)

print("✅ 模型已通过 UnSloth 极速加载！")
print(f"模型类型: {type(model).__name__}")""")

code("""# 给模型注入 LoRA 适配层
# 注意 target_modules：我们选择把 LoRA 贴在注意力头和 MLP 的关键投影矩阵上
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                        # LoRA 秩 (rank)，越大学习能力越强但显存越多
    target_modules=[             # 在哪些矩阵旁边贴便签条
        "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力头的四大矩阵
        "gate_proj", "up_proj", "down_proj",       # MLP 的三大矩阵
    ],
    lora_alpha=16,               # 缩放因子
    lora_dropout=0,              # Dropout 率
    bias="none",
    use_gradient_checkpointing="unsloth",  # 极度节省显存的梯度检查点
)

# 看看注入 LoRA 后，可训练参数变成了多少
model.print_trainable_parameters()
print("\\n🤯 看到了吗？几十亿的参数里，我们只需要训练不到 1% 的便签条参数！")""")

md("""---
## 4. 开始训练！

我们使用 Hugging Face 的 `SFTTrainer` (Supervised Fine-Tuning Trainer) 来进行训练。
由于只有 500 条数据、T4 显卡，我们训练 1 个 epoch 只需要几分钟。""")

code("""from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,        # batch size，T4 上 2 就够了
        gradient_accumulation_steps=4,        # 梯度累积以模拟更大的 batch
        warmup_steps=5,
        num_train_epochs=1,                   # 只跑 1 轮演示
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,                     # 每 10 步打印一次 loss
        output_dir="/content/lora_output",
        optim="adamw_8bit",                   # 8-bit 优化器再省一波显存
    ),
)

print("🚀 微调启动！请观察 loss 数值的下降趋势...")
trainer_stats = trainer.train()

print(f"\\n✅ 训练完成！最终 Loss: {trainer_stats.training_loss:.4f}")
print(f"⏱️ 总训练时间: {trainer_stats.metrics['train_runtime']:.1f} 秒")""")

md("""---
## 5. 效果验证：前后对比

激动人心的时刻！让我们用同一道医学问题，分别测试"原始 Qwen"和"植入记忆后的 Qwen"。""")

code("""# 切换到推理模式
FastLanguageModel.for_inference(model)

# 准备测验问题
test_question = "我最近经常头痛，尤其是在用眼过度后，这可能是什么原因？需要做什么检查？"

messages = [
    {"role": "system", "content": "你是一名专业的医学AI助手，请根据医学知识准确回答用户的健康问题。"},
    {"role": "user", "content": test_question}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

print("============ [植入记忆后的 Qwen] 微调版输出 ============")
outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))""")

md("""### 💡 极客彩蛋：Thinking Mode 切换

Qwen3.5 支持"思考模式"，模型会在 `<think>...</think>` 标签中展示推理过程。
如果你觉得它"思考太久"导致回复慢，可以通过参数关闭：

```python
# 在 apply_chat_template 中加入：
inputs = tokenizer.apply_chat_template(
    messages,
    ...,
    enable_thinking=False  # 关闭深度思考，直接用 Fast 模式秒答
)
```

⚠️ 注意：Qwen3.5 **不再支持** 旧版 Qwen3 的 `/think` `/nothink` 文字指令切换。""")

md("""---
## 第 3 阶段通关小结

你刚才完成了一件了不起的事情：
- 弄明白了为什么全参训练是不可能的，LoRA 是一种极其优雅的"外挂式"解决方案。
- 亲手构造了"饲料"数据集，并学会了把原始数据转换成聊天模板格式。
- 在 15GB 的 T4 显卡上，仅用几分钟就完成了一次 4-bit LoRA 微调！
- 见证了微调前后模型在专业问答上的差距。

下一阶段（第 4 阶段），我们将进入更加黑客级别的操作：用 PyTorch Hook 窃听大模型内部的隐藏状态，画出注意力热力图，甚至尝试直接注入控制向量来操纵模型的"人格"！ 🔥""")

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
