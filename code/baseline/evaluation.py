import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from datasets import load_dataset, load_metric
import evaluate
from torch.utils.data import DataLoader


def prediction():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 加载tokenizer和model
    tokenizer_path = "/root/autodl-tmp/Llama-2-7b-hf"
    model_path = "/root/autodl-tmp/hf_ckpt"  

    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    # 加载测试数据
    test_data = load_dataset('json', data_files={'test': '/root/RAG_NLIBench/data/test_datasets/Translation_200.json'})['test']
    
    # 准备输出文件
    output_data = []
    i = 1
    for data_point in test_data:
        # 通用prompt
        '''inputs = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
"""'''
        # Textual Entailment prompt
        '''inputs = f"""### Instruction
Task: Determine the relationship between the two sentences provided. Choose the appropriate letter to represent the relationship:
- E: Sentence 1 entails Sentence 2
- C: Sentence 1 contradicts Sentence 2
- N: The relationship is neutral (cannot be determined from the text alone)

Indicate your answer by choosing E, C, or N based on the relationship between the two sentences. All you need to do is output E, N, and C. # noqa: E501

### Input:
{data_point["input"]}

### Response:
"""'''
        '''inputs = f"""Below is an instruction that describes a Textual Entailment task. This task's goal is to determine the relationship between the two sentences provided, paired with an input that provides further context. You should choose the appropriate letter to represent the relationship.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
"""'''
        inputs = f"""Below is an instruction that describes a Textual Entailment task. This task's goal is to translate the provided sentence from its original language into English, paired with an input that provides further context. You should focus on accuracy and fluency in your translation. .  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
"""
        input_ids = tokenizer(inputs, return_tensors="pt")
        input_ids = input_ids["input_ids"].to(device)

        # 生成输出
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                num_beams=4,
                max_new_tokens=128, 
                eos_token_id=2,  
                bos_token_id=1,  
                pad_token_id=0
            )

        # 解码预测结果
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        try:
            #print(decoded_preds)
            decoded_preds = decoded_preds.split("\n")[-1].strip()
            data_point['predicted_output'] = decoded_preds
        except (IndexError, RuntimeError) as e:
            print(decoded_preds)
            print(f"捕获到异常：{e}")
            continue
        
        # 保存结果到数据结构
        output_data.append(data_point)
        i = i + 1
        print(i)

        # 保存修改后的数据集
        with open('/root/RAG_NLIBench/results/result-0428/Translation_200_1.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

def bleu(file_path):
    '''
    evaluate BLEU score
    '''
    bleu = load_metric("bleu")

    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    sum_score = 0
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref = [[item['output'].split(' ')]]
        pred = [item['predicted_output'].split(' ')]

        # 计算单个BLEU分数
        single_result = bleu.compute(predictions=pred, references=ref)
        single_bleu_score = single_result['bleu']
        sum_score = sum_score + single_bleu_score

        # 保存结果到数据结构
        item['BLEU'] = single_bleu_score
        output_data.append(item)
    
    # 保存修改后的数据集
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"Overall BLEU score for the dataset: {sum_score / len(output_data)}")

def rouge(file_path):
    rouge = load_metric("rouge")

    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    sum_rouge_1 = 0
    sum_rouge_2 = 0
    sum_rouge_l = 0
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref = [item['output']]
        pred = [item['predicted_output']]

        # 计算单个ROUGE分数
        single_result = rouge.compute(predictions=pred, references=ref)
        # ROUGE输出通常包含多个分数，如ROUGE-1, ROUGE-2, ROUGE-L
        rouge_1_score = single_result['rouge1'].mid.fmeasure
        sum_rouge_1 += rouge_1_score
        rouge_2_score = single_result['rouge2'].mid.fmeasure
        sum_rouge_2 += rouge_2_score
        rouge_l_score = single_result['rougeL'].mid.fmeasure
        sum_rouge_l += rouge_l_score

        # 保存结果到数据结构
        item['ROUGE_1'] = rouge_1_score
        item['ROUGE_2'] = rouge_2_score
        item['ROUGE_l'] = rouge_l_score
        output_data.append(item)
    
    # 保存修改后的数据集
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f'ROUGE_1: {sum_rouge_1 / len(output_data)}')
    print(f'ROUGE_2: {sum_rouge_2 / len(output_data)}')
    print(f'ROUGE_l: {sum_rouge_l / len(output_data)}')

def BERTScore(file_path):
    bertscore = load_metric("bertscore")

    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    sum_bertscore = 0
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref = [item['output']]
        pred = [item['predicted_output']]

        # 计算单个BERTScore
        single_result = bertscore.compute(predictions=pred, references=ref, lang="en")  
        f1 = single_result['f1']

        # 保存结果到数据结构
        item['BERTScore_f1'] = f1[0]
        sum_bertscore += f1[0]
        output_data.append(item)
    
    # 保存修改后的数据集
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f'BERTScore: {sum_bertscore / len(output_data)}')

def Acc():
    f1 = load_metric("f1")

    with open('/root/autodl-tmp/Textual_Entailment_snli_100_1.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    pred = []
    ref = []
    sum_acc = 0

    dict_map = {
        "E": 0,
        "C": 1,
        "N": 2
    }
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref.append(dict_map[item['output']])
        pred.append(dict_map[item['predicted_output']])

        # 计算单个F1
        if dict_map[item['output']] == dict_map[item['predicted_output']]:
            acc_score = 1.0
        else: 
            acc_score = 0.0

        # 保存结果到数据结构
        item['acc'] = acc_score
        sum_acc += acc_score
        output_data.append(item)
    
    # 保存修改后的数据集
    with open('/root/autodl-tmp/Textual_Entailment_snli_100_1.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    f1_score = f1.compute(predictions=pred, references=ref)
    print(f'Acc: {sum_acc / len(output_data)}')
    print(f1_score['f1'])

def exact_match():
    exact_match = load_metric("exact_match")

    with open('/root/RAG_NLIBench/results/result-0428/Textual_Entailment_snli_100_2.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    ref = []
    pred = []
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref.append(item['output'])
        pred.append(item['predicted_output'])
    
    results = exact_match.compute(references=ref, predictions=pred,ignore_case=True, ignore_punctuation=True)

    print(round(results["exact_match"], 2))

if __name__ == '__main__':
    prediction()
    #exact_match()
    file_path = '/root/RAG_NLIBench/results/result-0428/Translation_200_1.json'
    bleu(file_path)
    rouge(file_path)
    BERTScore(file_path)

