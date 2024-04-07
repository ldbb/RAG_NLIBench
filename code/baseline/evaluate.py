import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

class CustomDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length
        with open(filename, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append((item["instruction"], item["input"], item["output"], item["category"]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instruction, input_text, output_text, category = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            instruction + " " + input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(output_text, dtype=torch.long)
        }

# 定义tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# 加载数据
filename = 'your_dataset.jsonl'
dataset = CustomDataset(filename, tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

# 假设你已经有了一个训练好的模型
# 这里是用模型进行预测的示例
model.eval()
predictions = []
labels = []
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels += batch['labels'].numpy().tolist()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions += torch.argmax(logits, dim=-1).numpy().tolist()

# 计算F1分数
f1 = f1_score(labels, predictions, average='weighted')
print(f"F1 Score: {f1}")


