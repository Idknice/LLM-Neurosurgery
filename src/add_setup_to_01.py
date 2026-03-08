import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\01_pytorch_huggingface_basics.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# The first cell is the intro markdown. We will insert the setup cells right after it.
setup_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 0. 课前准备 (极其重要！)\n",
            "\n",
            "**请注意！Colab 的每一个 Notebook 实例都是独立的“虚拟机”。** \n",
            "你在上一个 Notebook (如 `00`) 中安装的包、挂载的 Google Drive，在打开新的 Notebook 时都会统统归零。\n",
            "\n",
            "所以在每一次开启新的挑战前，请养成好习惯，执行以下单元格把神级兵器和硬盘装配好！"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. 安装我们在解剖过程中必需的屠龙刀\n",
            "# 包括前沿的 transformers 从而支持最新的视觉多模态大模型框架\n",
            "!pip install -q -U torch transformers accelerate datasets huggingface_hub bitsandbytes qwen-vl-utils git+https://github.com/huggingface/transformers.git"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from google.colab import drive\n",
            "import os\n",
            "\n",
            "# 2. 挂载持久化硬盘\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# 3. 指定之前我们在 00 阶段下载模型的地方，这样接下来的黑科技就不会再重复下载庞大的 10GB 权重文件了！\n",
            "cache_dir = \"/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache\"\n",
            "model_id = \"Qwen/Qwen3.5-4B\"\n",
            "\n",
            "print(\"✅ 工作区与缓存目录已绑定完毕，可以开始解剖！\")"
        ]
    }
]

# Insert after the first cell (index 1)
data['cells'] = [data['cells'][0]] + setup_cells + data['cells'][1:]

# In 01 notebook, we need to ensure people don't define model_id and cache_dir again incorrectly. 
# In the original 01, they were assigned inside the AutoProcessor block. I'll remove the redundant declarations from that block to avoid confusion.
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = [line for line in source if 'model_id = ' not in line and 'cache_dir = ' not in line]
        cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
    f.write("\n")
