import json
import os
import random
from sklearn.model_selection import train_test_split

# 设置根目录和目标文件数
root_dir = '/root/autodl-tmp/Instruction-tuning_Datasets/train_data'  
total_records_to_sample = 62500  # 总共要随机选择的记录数量
test_ratio = 0.2  # 测试集比例
output_test_file = '/root/autodl-tmp/Instruction-tuning_Datasets/test_dataset_1.json'
output_train_file = '/root/autodl-tmp/Instruction-tuning_Datasets/train_dataset_1.json'

# 遍历根目录下的所有jsonl文件
def get_all_jsonl_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jsonl'):
                yield os.path.join(root, file)

# 从每个文件中读取所有记录
def read_records(files):
    all_records = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                try:
                    data = {
                        "instruction": json_data['instruction'],
                        "input": json_data['input'],
                        "output": json_data['output'],
                        "category": json_data['category']
                    }
                    all_records.append(data)
                except KeyError as e:
                    print(f"Missing key {e} in file: {file}")
                    continue
    return all_records

# 获取所有记录并进行数据划分
files = list(get_all_jsonl_files(root_dir))
all_records = read_records(files)

# 随机选取指定数量的记录，如果总记录数小于目标数量，则返回所有记录
if len(all_records) > total_records_to_sample:
    selected_records = random.sample(all_records, total_records_to_sample)
else:
    selected_records = all_records

# 将选取的数据按照80:20比例分割为训练集和测试集
train_records, test_records = train_test_split(selected_records, test_size=test_ratio, random_state=42)

# 将训练集和测试集数据保存为JSON文件
with open(output_train_file, 'w', encoding='utf-8') as f:
    json.dump(train_records, f, ensure_ascii=False, indent=2)

with open(output_test_file, 'w', encoding='utf-8') as f:
    json.dump(test_records, f, ensure_ascii=False, indent=2)

print(f"完成！已保存训练集{len(train_records)}条记录到{output_train_file}。")
print(f"完成！已保存测试集{len(test_records)}条记录到{output_test_file}。")
