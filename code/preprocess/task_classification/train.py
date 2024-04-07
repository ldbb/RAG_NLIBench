# 训练一个MLP分类器
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 加载数据
data = []
with open('Instruction-tuning_Datasets/train_data/Super-Natural Instruction/task_classification/train.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        json_data = json.loads(line)
        data.append(json_data)

# 提取instruction和标签
texts = [item['instruction'] for item in data]
labels = [item['category'] for item in data]

# 使用TF-IDF转换文本
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 编码标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 构建MLP分类器
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4,
                    solver='adam', verbose=10, random_state=1,
                    learning_rate_init=.001)

# 训练模型
mlp.fit(X_train, y_train)

# 评估模型
predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))

# 保存模型
with open('code/task_classification/model.pkl', 'wb') as file:
    pickle.dump(mlp, file)

# 保存特征化向量对象
with open('code/task_classification/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# 保存标签编码器
with open('code/task_classification/label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

