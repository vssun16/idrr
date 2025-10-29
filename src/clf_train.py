import os
# 设置可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch
import random
import numpy as np
import pandas as pd

from typing import Dict
from pathlib import Path as path
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers import (Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer)

from IDRR_data import IDRRDataFrames

# 获取当前文件所在的目录和根目录
SRC_DIR = path(__file__).parent
ROOT_DIR = SRC_DIR.parent
SEED = 42
# MODEL_NAME = 'Qwen3-0.6B'
MODEL_NAME = 'flan-t5-base'
MODEL_NAME = 'reberta-base'
EPOCH = 5
EXP_ID = 1
OUTPUT_DIR = f'{ROOT_DIR}/results/clf/{MODEL_NAME}/epo{EPOCH}/{EXP_ID}'

batch_size = 16
learning_rate = 5.0e-5
warmup_ratio = 0.1

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# === dataset ===
class CustomDataset(Dataset):
    def __init__(self, df, label_list, tokenizer) -> None:
        self.df:pd.DataFrame = df
        label_num = len(label_list)
        self.id2label = {i:label for i, label in enumerate(label_list)}
        self.ys = np.eye(label_num, label_num)[self.df['label11id']]
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        model_inputs = self.tokenizer(
            row['arg1'], row['arg2'],
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=512,
        )
        model_inputs['labels'] = self.ys[index]
        return model_inputs
    
    def __len__(self):
        return self.df.shape[0]

# === metric ===
class ComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.metric_names = ['Macro-F1', 'Acc']
    
    def __call__(self, eval_pred):
        """
        n = label categories
        eval_pred: (pred, labels)
        # pred: np.array [datasize, ]
        pred: np.array [datasize, n]
        labels: np.array [datasize, n]
        X[p][q]=True, sample p belongs to label q (False otherwise)
        """
        pred, labels = eval_pred
        pred: np.ndarray
        labels: np.ndarray
        
        pred = pred[..., :len(self.label_list)]
        labels = labels[..., :len(self.label_list)]
        
        # pred = pred!=0
        max_indices = np.argmax(pred, axis=1)
        bpred = np.zeros_like(pred, dtype=int)
        bpred[np.arange(pred.shape[0]), max_indices] = 1
        pred = bpred
        assert ( pred.sum(axis=1)<=1 ).sum() == pred.shape[0]
        labels = labels!=0
        
        res = {
            'Macro-F1': f1_score(labels, pred, average='macro', zero_division=0),
            'Acc': np.sum(pred*labels)/len(pred),
        } 
        return res

# === callback ===
class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        log_filepath=None,
    ):
        super().__init__()
        if log_filepath:
            self.log_filepath = log_filepath
        else:
            self.log_filepath = path(OUTPUT_DIR) / 'log.jsonl'
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.log_filepath, 'a', encoding='utf8')as f:
            f.write(str(kwargs['logs'])+'\n')

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        pass

def main():
    set_seed(SEED)
    
    # === data ===
    dfs = IDRRDataFrames(
        data_name='pdtb2',
        data_level='top',
        data_relation='Implicit',
        data_path='/data/sunwh/idrr/data/raw/pdtb2.p1.csv',
    )
    label_list = dfs.label_list
    
    print(len(label_list))

    # === model ===
    model_name_or_path = f'/data/sunwh/pretrained_models/{MODEL_NAME}'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 确保 pad_token 设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型时确保 pad_token_id 正确设置
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, 
        num_labels=len(label_list),
        pad_token_id=tokenizer.pad_token_id
    )

    # === args ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        run_name='',

        # strategies of evaluation, logging, save
        eval_strategy = "epoch",
        eval_steps = 1,
        logging_strategy = 'steps',
        logging_steps = 10,
        # save_strategy = 'steps',
        # save_steps = 10,
        # max_steps=2,
        
        # optimizer and lr_scheduler
        optim = 'adamw_torch',
        # optim = 'sgd',
        learning_rate = learning_rate,
        # weight_decay = 1.0e-6,
        lr_scheduler_type = 'linear',
        warmup_ratio = warmup_ratio,
        
        # epochs and batches 
        num_train_epochs = EPOCH,
        # max_steps = args.max_steps,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        # gradient_accumulation_steps = 4,  # 从8减少到4
        
        # train consumption - 内存优化
        eval_accumulation_steps=4,  # 从8减少到4
        bf16=False,
        fp16=True,
        
        gradient_checkpointing=True,  # 启用梯度检查点
        dataloader_pin_memory=False,  # 禁用pin memory以节省内存
        remove_unused_columns=True,  # 移除未使用的列
        # 添加内存优化选项
        ddp_find_unused_parameters=False,
    )

    # 加载训练集、验证集和测试集
    train_dataset = CustomDataset(dfs.train_df, label_list, tokenizer)
    dev_dataset = CustomDataset(dfs.dev_df, label_list, tokenizer)
    test_dataset = CustomDataset(dfs.test_df, label_list, tokenizer)

    # === train ===
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_metrics=ComputeMetrics(dfs.label_list),
        callbacks=[CustomCallback()],
    )

    try:
        # 开始训练和评估
        train_result = trainer.train()
        test_result = trainer.evaluate(eval_dataset=test_dataset)
        print(f'> train_result:\n  {train_result}')
        print(f'> test_result:\n  {test_result}')
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU内存不足！建议进一步减少batch_size或max_length")
            print(f"当前配置: batch_size={batch_size}, max_length=256")
            print("建议尝试: batch_size=2, max_length=128")
        raise e
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pass

if __name__ == '__main__':
    main()