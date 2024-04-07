import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from models.transformer import TransformerModel
from utils.dataloader import CustomDataset
from sklearn.metrics import accuracy_score
import numpy as np

# 参数设置
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 10  # 早停的耐心值
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_MODEL_PATH = 'model_best.pt'

# 加载数据
train_dataset = CustomDataset('data/train_data.jsonl')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 模型初始化
model = TransformerModel().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 早停和最佳模型保存
best_accuracy = 0.0
patience_counter = 0

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    predictions, true_labels = [], []
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 预测
        _, predicted = torch.max(outputs.data, 1)
        predictions += predicted.cpu().numpy().tolist()
        true_labels += labels.cpu().numpy().tolist()

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}")

    # 保存最好的模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Saved Best Model")
        patience_counter = 0  # 重置早停计数器
    else:
        patience_counter += 1
    
    # 检查早停
    if patience_counter >= PATIENCE:
        print("Early stopping triggered")
        break

print("Training completed.")
