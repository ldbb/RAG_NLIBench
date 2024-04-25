import json
import os
from joblib import load

def process_Super_Natural_Instruction():
    # transform Super Natural Instruction datasets 
    file_dir = './Super-Natural Instruction'

    count = 0
    data_list = []
    for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    for key in ['Positive Examples', 'Instances']:
                        for item in json_data[key]:
                            if isinstance(item['output'], list):
                                item['output'] = item['output'][0]
                            data = {
                                "instruction": json_data['Definition'][0],
                                "input": item['input'],
                                "output": item['output'],
                                "category": json_data['Categories'][0]
                            }
                            data_list.append(data)

                            if len(data_list) == 50:
                                count_str = str(count)
                                file_name = 'train_' + count_str + '.json'
                                with open(os.path.join('/root/autodl-tmp/Instruction-tuning_Datasets/test_datasets/bleu_rouge_bertscore', file_name), 'w', encoding='utf-8') as file:
                                    for item in data_list:
                                        json_string = json.dumps(item, ensure_ascii=False)
                                        file.write(json_string + '\n')
                                count += 1
                                data_list = []


def process_Dolley():
    # transform Dolley datasets
    file_dir = './Dolley'
    data_list = []

    for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            json_data = json.loads(line)
                            
                            data = {
                                "instruction": json_data['instruction'],
                                "input": json_data['context'],
                                "output": json_data['response'],
                                "category": json_data['category']
                            }

                            data_list.append(data)
                    file_name = file.split('.')[0] + '_train.jsonl'
                    with open(os.path.join('./Dolley', file_name), 'w', encoding='utf-8') as file:
                        for item in data_list:
                            json_string = json.dumps(item, ensure_ascii=False)
                            file.write(json_string + '\n')
                            
def process_alpaca_GPT4_LLM():
    # transform Alpaca and GPT-4-LLM datasets
    data_list = []

    with open('raw_datasets/GPT-4-LLM/alpaca_gpt4_data.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for item in json_data:  
        data = {
            "instruction": item['instruction'],
            "input": item['input'],
            "output": item['output'],
            "category": ""
        }

        data_list.append(data)
    with open('raw_datasets/GPT-4-LLM/train.jsonl', 'w', encoding='utf-8') as file:
        for item in data_list:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')

def process_self_instruct():
    # tramsform self-instruct datasets
    data_list = []

    with open('./Self-instruct/all_instances_82K.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line)
                            
            data = {
                "instruction": json_data['instruction'],
                "input": json_data['input'],
                "output": json_data['output'],
                "category": ""
            }
            data_list.append(data)
    with open('./Self-instruct/train.jsonl', 'w', encoding='utf-8') as file:
        for item in data_list:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n')                 

def process_unnatural_instruction():
    # transfrom Unnatural instructiton datasets
    file_dir = 'raw_datasets/Unnatural-instructions/full_data.jsonl'
    
    data_list = []
    with open(file_dir, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)

            for item in json_data['instances']:
                data = {
                    "instruction": json_data['instruction'],
                    "input": item['input'],
                    "output": item['output'],
                    "category": ""
                }
                data_list.append(data)
            
            if "reformulations" in json_data:
                for item in json_data['reformulations']:
                    data = {
                        "instruction": item['instruction'],
                        "input": item['input'],
                        "output": item['output'],
                        "category": ""
                    }
                    data_list.append(data)

    with open('raw_datasets/Unnatural-instructions/train.jsonl','w', encoding='utf-8') as file:
        for item in data_list:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + '\n') 

def task_classification():
    # 有些数据集没有按NLP任务分类，这里对数据集进行分类

    file_dir = 'raw_datasets/Unnatural-instructions'
    # 加载模型
    classifier = load('code/task_classification/classifier.joblib')
    # 加载特征化向量对象
    vectorizer = load('code/task_classification/vectorizer.joblib')

    # 预测
    text = []
    with open(os.path.join(file_dir, 'train.jsonl'), 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            text.append(item['instruction'])

    # 使用之前保存的向量化对象转换文本
    X_new = vectorizer.transform(text)
        
    # 使用加载的模型进行预测
    predictions = classifier.predict(X_new)

    with open(os.path.join(file_dir, 'train.jsonl'), 'r', encoding='utf-8') as input_file, \
        open(os.path.join(file_dir, 'train_category.jsonl'), 'w', encoding='utf-8') as output_file:
            for line, prediction in zip(input_file, predictions):
                item = json.loads(line)
                # 假设你想要将预测结果保存到'category'字段
                item['category'] = prediction
                # 将修改后的记录写回到新的jsonl文件
                output_file.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    process_alpaca_GPT4_LLM()

