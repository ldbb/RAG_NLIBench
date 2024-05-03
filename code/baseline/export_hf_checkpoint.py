# 合并模型权重
import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

# 设置基本模型路径和微调模型的保存路径
BASE_MODEL = "/root/autodl-tmp/Llama-2-7b-hf"
FINETUNED_MODEL_DIR = "/root/RAG_NLIBench/code/baseline/experiments_2"  # 微调模型的输出目录

# 加载Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

# 加载原始预训练模型
base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)
lora_model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_MODEL_DIR,
    torch_dtype=torch.float16,
)

# 执行合并操作
lora_model = lora_model.merge_and_unload()
lora_model.eval()

# 提取合并后的状态字典
lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

# 保存合并后的模型
LlamaForCausalLM.save_pretrained(
    base_model,
    "root/autodl-tmp/hf_ckpt",
    state_dict=deloreanized_sd,
    max_shard_size="400MB"
)
