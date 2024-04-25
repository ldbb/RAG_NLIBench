import matplotlib.pyplot as plt
import json

# 假设log_history是你从JSON文件中读取的数据列表
with open('trainer_state.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
log_history = json_data['log_history']  # 这里应该包含你的数据

# 初始化列表以存储步骤、训练损失和测试损失
steps = []
train_losses = []
test_losses = []

# 遍历记录，分离出训练和测试损失
for entry in log_history:
    steps.append(entry['step'])
    if 'loss' in entry:  # 训练损失
        train_losses.append(entry['loss'])
    if 'eval_loss' in entry:  # 测试损失
        test_losses.append(entry['eval_loss'])

# 确保步骤和损失列表长度一致，对于测试损失，可能需要调整以对齐步骤
train_steps = [entry['step'] for entry in log_history if 'loss' in entry]
test_steps = [entry['step'] for entry in log_history if 'eval_loss' in entry]

# 创建图表
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_losses, label='Training Loss')
plt.plot(test_steps, test_losses, label='Test Loss', linestyle='--')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Training and Test Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

# 显示图表
plt.show()