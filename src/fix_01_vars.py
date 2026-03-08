import json
import os

filepath = r'c:\Users\golde\code\LLM-Neurosurgery\notebooks\01_pytorch_huggingface_basics.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('from transformers import AutoProcessor' in line for line in source):
            # Insert the assignments before AutoProcessor.from_pretrained
            insert_idx = 0
            for i, line in enumerate(source):
                if 'processor = AutoProcessor.from_pretrained' in line:
                    insert_idx = i
                    break
            
            # If not already present
            if not any('model_id =' in line for line in source):
                source.insert(insert_idx, 'model_id = "Qwen/Qwen3.5-4B"\n')
                source.insert(insert_idx+1, 'cache_dir = "/content/drive/MyDrive/LLM_Neurosurgery/huggingface_cache"\n\n')
            
            cell['source'] = source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)
    f.write("\n")
