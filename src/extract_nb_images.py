import json
import base64
import os
import glob
from pathlib import Path

def extract_attachments(notebook_path, assets_dir):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nbv_data = json.load(f)

    changed = False
    nb_name = Path(notebook_path).stem

    for idx, cell in enumerate(nbv_data.get('cells', [])):
        if 'attachments' in cell:
            attachments = cell['attachments']
            for att_name, att_data in attachments.items():
                for mime_type, b64_data in att_data.items():
                    # 确定扩展名
                    ext = mime_type.split('/')[-1]
                    if ext == 'jpeg':
                        ext = 'jpg'
                    
                    # 生成新文件名和路径
                    new_filename = f"{nb_name}_cell{idx}_{att_name}"
                    if not new_filename.endswith(f".{ext}"):
                        new_filename += f".{ext}"
                        
                    save_path = os.path.join(assets_dir, new_filename)
                    
                    # 解码并保存图片
                    img_data = base64.b64decode(b64_data)
                    with open(save_path, 'wb') as img_f:
                        img_f.write(img_data)
                    print(f"Extracted: {save_path}")

                    # 修改 Markdown 源码中的引用
                    # 从 ![alt](attachment:image.png) 修改为 ![alt](../assets/images/new_filename)
                    if 'source' in cell:
                        new_source = []
                        for line in cell['source']:
                            new_line = line.replace(f"attachment:{att_name}", f"../assets/images/{new_filename}")
                            new_source.append(new_line)
                        cell['source'] = new_source
            
            # 删除附件，减轻 notebook 体积
            del cell['attachments']
            changed = True

    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nbv_data, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Updated notebook: {notebook_path}")

if __name__ == '__main__':
    project_root = r"c:\Users\golde\code\LLM-Neurosurgery"
    notebooks_dir = os.path.join(project_root, "notebooks")
    assets_dir = os.path.join(project_root, "assets", "images")
    
    os.makedirs(assets_dir, exist_ok=True)
    
    notebook_files = glob.glob(os.path.join(notebooks_dir, "*.ipynb"))
    for nb_file in notebook_files:
        extract_attachments(nb_file, assets_dir)
