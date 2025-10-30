import os
os.environ['CUDA_VISIBLE_DEVICES']='6'

import torch
import random
import numpy as np
import json
import swanlab
swanlab.init(mode='disabled')
from transformers import LlamaForCausalLM

from typing import Dict, List
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

SEED = 42
WORK = "baseline"
# SEPARATOR_TOKEN = "</think>"
PAD_TOKEN = "<|finetune_right_pad_id|>" # for llama to pad
RESPONSE_TEMPLATE = "<|end_header_id|>\n\n"

MODEL_ID = "/data/sunwh/pretrained_models/Meta-Llama-3.1-8B-Instruct"
# MODEL_ID = "/data/sunwh/pretrained_models/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR = f"./expt/{WORK}/sft_debug/"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    ## 设置随机种子
    set_seed(SEED)
    print(f'Setting random seed: {SEED}\n')
    
    ## 设置分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        print(f"Setting pad_token to {PAD_TOKEN}")
        tokenizer.pad_token = PAD_TOKEN

    tokenizer.padding_side = "right"
    # separator_token_id = tokenizer.convert_tokens_to_ids(SEPARATOR_TOKEN)
    print(f"Tokenizer pad token: {tokenizer.pad_token}")
    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")
    # print(f"Separator token: {SEPARATOR_TOKEN}")
    # print(f"Separator token ID: {separator_token_id}")
    print(f"Tokenizer vocab size: {len(tokenizer)}\n")

    # 使用 ChatML collator，仅对补全部分计算损失（prompt 部分标签为 -100）
    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,response_template=RESPONSE_TEMPLATE)


    ## 加载数据
    print('******loading dateset ...******')
    dataset = load_dataset(
        "json",
        data_files={"train": f"/data/sunwh/idrr/data/{WORK}/pdtb2_train.json", 
                    "validation": f"/data/sunwh/idrr/data/{WORK}/pdtb2_dev.json",
                    # "test": f"./data/{WORK}/pdtb2_test.json"
                    },
    )

    def format_fn(example: dict):
        return {
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
        }
    dataset = dataset.map(format_fn, num_proc=16, remove_columns=dataset['train'].column_names)
    print(dataset)
    print(dataset['train'][0])
    print('******loading dateset done!!!******\n')

    ## 定义LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    ## 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # 使用fp16
        device_map={"": Accelerator().local_process_index},
    )
    print(f"Generation config: {model.generation_config}\n")
    
    # print('*'*6 + 'Trainable parameters' + '*'*6)
    # model.print_trainable_parameters() # 打印可训练参数信息
    # print('*'*32 + '\n')
    ## 训练配置
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=0.2,
        save_steps=0.2,
        logging_steps=10,
        learning_rate=5e-5 ,
        fp16=True,  # 启用半精度训练
        save_strategy="steps",
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        # report_to="tensorboard",
        # save_safetensors=True,
        ddp_find_unused_parameters=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        seed=SEED,
    )

    ## SFTTrainer by trl
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        peft_config=lora_config,
    )
    # --- 开始训练 ---
    print("Starting training...")
    trainer.train()

    # --- 保存最终训练结果 ---
    print("Saving final training results...")
    trainer.save_model(OUTPUT_DIR) # Saves only the LoRA adapter weights
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()