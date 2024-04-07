# 使用加载完毕的模型对新的数据进行预测
import pickle
import json

# 加载模型
with open('code/task_classification/model.pkl', 'rb') as file:
    model = pickle.load(file)

# 加载特征化向量对象
with open('code/task_classification/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# 加载标签编码器
with open('code/task_classification/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# 加载数据
data = []
with open('Instruction-tuning_Datasets/train_data/Unnatural-instructions/train_category.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        json_data = json.loads(line)
        data.append(json_data)

# 提取instruction
texts = [item['instruction'] for item in data]

# 使用TF-IDF转换文本
X = vectorizer.transform(texts)

# 预测
predictions = model.predict(X)

# 解码标签
labels = label_encoder.inverse_transform(predictions)

# 保存预测结果
with open('Instruction-tuning_Datasets/train_data/Unnatural-instructions/train_category.jsonl', 'w', encoding='utf-8') as file:
    for item, label in zip(data, labels):
        item['category'] = label
        json_string = json.dumps(item, ensure_ascii=False)
        file.write(json_string + '\n')

