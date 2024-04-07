import json
import os
import random
from sklearn.model_selection import train_test_split

def preprocess_data(jsonl_files, test_size=0.2, seed=42):
    """
    读取多个jsonl格式的数据文件，混合并划分为训练集和测试集。
    
    参数:
    - jsonl_files: jsonl文件列表
    - test_size: 测试集比例
    - seed: 随机种子，确保可重复性
    
    返回:
    - train_data, test_data: 分割后的训练数据和测试数据
    """
    data = []
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    
    # 打乱数据
    random.seed(seed)
    random.shuffle(data)
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    
    return train_data, test_data

def save_data(data, file_path):
    """
    将处理后的数据保存为jsonl文件。
    
    参数:
    - data: 要保存的数据列表
    - file_path: 保存文件的路径
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def main():
    # 定义jsonl文件路径列表

    file_dir = 'Instruction-tuning_Datasets/train_data'
    jsonl_files = []  # 添加jsonl文件路径
    for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
   
    # 预处理数据
    train_data, test_data = preprocess_data(jsonl_files)
    
    # 保存处理后的数据
    save_data(train_data, 'code/baseline/train_data.jsonl')
    save_data(test_data, 'code/baseline/test_data.jsonl')
    print(f"Preprocessing completed. Train data: {len(train_data)}, Test data: {len(test_data)}")

if __name__ == '__main__':
    main()
