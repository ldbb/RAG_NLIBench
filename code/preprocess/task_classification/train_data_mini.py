# 生成小批量训练样本，每类选取100个数据样本，并划分训练集和测试集，比例为8:2
import json
import os

file_dir = 'Instruction-tuning_Datasets/raw_datasets/Super-Natural Instruction'

data_list = []
for root, dirs, files in os.walk(file_dir):
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                    
                data = {
                    "instruction": json_data['Definition'][0],
                    "category": json_data['Categories'][0]
                }
                data_list.append(data)                          

print(len(data_list))
with open('Instruction-tuning_Datasets/train_data/Super-Natural Instruction/task_classification/train.jsonl', 'w', encoding='utf-8') as file:
    for item in data_list:
        json_string = json.dumps(item, ensure_ascii=False)
        file.write(json_string + '\n')
                                
                                