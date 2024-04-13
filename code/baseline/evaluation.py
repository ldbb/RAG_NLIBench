import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer_path = "/root/autodl-tmp/Llama-2-7b-hf"
model_path = "/root/autodl-tmp/Llama-2-7b-hf"  # 换为合并后的路径

# 加载模型和tokenizer
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, load_in_8bit=True, device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

# 载入BLEU度量
bleu = load_metric("bleu")

# 假设您的测试数据已经预处理并保存为一个数据集
test_data = load_dataset('json', data_files={'test': 'path/to/your/test_data.jsonl'})['test']

def prepare_input(data_point):
    # 根据您的需求组合instruction和input字段
    combined_input = f"{data_point['instruction']} {data_point['input']}"
    return combined_input

# 对测试数据进行预处理
def preprocess_data(batch):
    # 准备模型输入
    inputs = tokenizer(batch['input'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 准备参考输出
    references = [[ref] for ref in batch['output']]
    
    return inputs, references

model.eval()
predictions, references = [], []
for data_point in test_data:
    model_input, ref = preprocess_data(prepare_input(data_point))
    outputs = model.generate(**model_input)
    
    # 将生成的token ids转换为文本
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 收集预测和参考文本
    predictions.extend(decoded_preds)
    references.extend(ref)

# 计算BLEU分数
results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score: {results['bleu']}")
