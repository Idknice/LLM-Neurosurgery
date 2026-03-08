# GitHub 与 Colab 联动指南

本指南将帮助你在 Google Colab 和 GitHub 之间丝滑地同步代码，确保持续开发不会丢失进度。

## 1. 从 GitHub 导入项目到 Colab
最直接的方法是直接在 Colab 中打开你的 GitHub Notebook：
1. 打开 [Google Colab](https://colab.research.google.com/)
2. 在弹出的窗口中选择 **GitHub** 标签页。
3. 授权 Colab 访问你的 GitHub 账号。
4. 搜索或输入你的仓库名称，例如 `<你的用户名>/LLM-Neurosurgery`。
5. 在下拉列表中选择对应的 Branch 和具体的 Notebook 文件打开。

> **小贴士**: 如果你在看 GitHub 上的 `.ipynb` 文件，也可以把 URL 中的 `github.com` 改成 `colab.research.google.com/github` 来快速在 Colab 打开。

## 2. 克隆仓库与配置环境 (推荐深入开发者使用)
如果你的代码需要互相调用，或读取本地的脚本，建议整体克隆仓库。
在 Colab Notebook 第一个单元格中运行：

```bash
# 为了在私有仓库工作，请替换你的 GitHub Personal Access Token (PAT)
# 注意：正式运行时绝对不要把带有 PAT 的代码直接公开保存！也可以用 Colab Secrets 安全存储。
!git clone https://<YOUR_PAT>@github.com/<YOUR_USERNAME>/LLM-Neurosurgery.git
%cd LLM-Neurosurgery

# 在这里我们也可以安装 uv，尽管 Colab 环境已经预装了很多大模型包
!curl -LsSf https://astral.sh/uv/install.sh | sh
!/root/.local/bin/uv pip install -r pyproject.toml # 或手动安装依赖
# 或者直接用 pip：
!pip install torch transformers peft trl accelerate bitsandbytes datasets
```

## 3. 保存更改回 GitHub
在 Colab 中修改了 Notebook 后，有两种方式能够存回 GitHub。

### 方式一 (原生保存)
1. 点击 Colab 菜单栏的 **File (文件)**。
2. 选择 **Save a copy in GitHub (在 GitHub 中保存副本)**。
3. 选择对应的仓库、分支，并填写 Commit 信息。

### 方式二 (命令行推送，适用于多文件修改)
如果你在克隆的文件夹里修改了多个文件（不仅是 Notebook，还有 `.py` 脚本）：
```bash
%cd /content/LLM-Neurosurgery
# 配置 git 用户信息
!git config --global user.email "your.email@example.com"
!git config --global user.name "Your Name"

# 提交并推送
!git add .
!git commit -m "Update from Colab"
!git push origin main
```

## 4. 持久化存储数据 (Google Drive 挂载)
由于 Colab 实例会在一定时间无活动后重置，所有的文件都会丢失，对于巨大的模型权重或数据集，必须保存在 Google Drive。

```python
from google.colab import drive
drive.mount('/content/drive')
```
模型和数据存放路径建议设置为 `/content/drive/MyDrive/LLM_data`。

## 5. Standard Operation Procedure (SOP): 下载大型开源模型并持久化到 GDrive
**假设你要运行一个 10GB 的 Qwen3.5 模型:**
1. 挂载 Google Drive
2. 在 GDrive 内创建一个永久保存权重的文件夹。
3. 利用 Hugging Face 官方工具下载到指定目录，防止每次启动重复下载。

```python
import os
from huggingface_hub import snapshot_download

# 指定模型保存路径在你的云盘中
cache_dir = "/content/drive/MyDrive/LLM_data/huggingface_models"
os.makedirs(cache_dir, exist_ok=True)

# 下载模型并使用刚才指定的 cache_dir
model_id = "Qwen/Qwen1.5-4B-Chat"
snapshot_download(repo_id=model_id, cache_dir=cache_dir)
```
这样，就算下次实例重置，只要再次挂载 Drive 并且指定同一个 `cache_dir`，Hugging Face transformers 库就会自动加载已存在的模型。
