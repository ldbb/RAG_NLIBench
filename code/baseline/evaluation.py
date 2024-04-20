import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from datasets import load_dataset, load_metric
import evaluate
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader


def prediction():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 加载tokenizer和model
    tokenizer_path = "/root/autodl-tmp/Llama-2-7b-hf"
    model_path = "/root/autodl-tmp/hf_ckpt"  

    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    # 加载测试数据
    test_data = load_dataset('json', data_files={'test': '/root/autodl-tmp/Instruction-tuning_Datasets/test_dataset.json'})['test']
    print(test_data)
    # 准备输出文件
    output_data = []
    i = 1
    for data_point in test_data:
        # 准备输入
        inputs = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
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
            decoded_preds = decoded_preds.split("### Response:")[1].strip()
            data_point['predicted_output'] = decoded_preds
            i = i + 1
        except (IndexError, RuntimeError) as e:
            print(f"捕获到异常：{e}")
            continue
        
        # 保存结果到数据结构
        output_data.append(data_point)
        print(i)
        if i == 2000:
            break

    # 保存修改后的数据集
    with open('/root/autodl-tmp/Instruction-tuning_Datasets/test_dataset_with_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def bleu():
    bleu = load_metric("bleu")

    with open('/root/RAG_NLIBench/translation.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref = [item['output']]
        pred = [item['predicted_output']]

        # 计算单个BLEU分数
        single_result = bleu.compute(predictions=pred, references=ref)
        single_bleu_score = single_result['bleu']
        sum_score = sum_score + single_bleu_score

        # 保存结果到数据结构
        item['bleu_score'] = single_bleu_score
        output_data.append(item)
    
    # 保存修改后的数据集
    with open('/root/RAG_NLIBench/translation_1.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"Overall BLEU score for the dataset: {overall_bleu_score}")

def rouge_evaluation():
    rouge = load_metric("rouge")

    with open('/root/RAG_NLIBench/translation_1.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref = sent_tokenize(item['output'])
        pred = sent_tokenize(item['predicted_output'])

        # 计算单个ROUGE分数
        single_result = rouge.compute(predictions=pred, references=ref)
        # ROUGE输出通常包含多个分数，如ROUGE-1, ROUGE-2, ROUGE-L
        rouge_1_score = single_result['rouge1'].mid.fmeasure  # 使用F1分数
        rouge_2_score = single_result['rouge2'].mid.fmeasure
        rouge_l_score = single_result['rougeL'].mid.fmeasure

        # 保存结果到数据结构
        item['rouge_1_score'] = rouge_1_score
        item['rouge_2_score'] = rouge_2_score
        item['rouge_l_score'] = rouge_l_score
        output_data.append(item)
    
    # 保存修改后的数据集
    with open('/root/RAG_NLIBench/translation_1.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def evaluate_with_bertscore():
    bertscore = load_metric("bertscore")

    with open('/root/RAG_NLIBench/translation_1.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    output_data = []
    for item in json_data:
        # 准备单个参考文本和预测文本
        ref = sent_tokenize(item['output'])
        pred = sent_tokenize(item['predicted_output'])

        # 计算单个BERTScore
        single_result = bertscore.compute(predictions=pred, references=ref, lang="en")  # 语言代码根据数据集语言调整
        precision = single_result['precision']
        recall = single_result['recall']
        f1 = single_result['f1']

        # 保存结果到数据结构
        item['bertscore_precision'] = precision[0]  # BERTScore输出是每个预测的列表
        item['bertscore_recall'] = recall[0]
        item['bertscore_f1'] = f1[0]
        output_data.append(item)
    
    # 保存修改后的数据集
    with open('/root/RAG_NLIBench/translation_1.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    prediction()