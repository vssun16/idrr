from IDRR_data import *

import os
import numpy as np
import pandas as pd
from pathlib import Path as path

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainerState, TrainerControl

# 设置可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

# 获取当前文件所在的目录和根目录
SRC_DIR = path(os.getcwd())
ROOT_DIR = SRC_DIR.parent

# === dataset ===
class T5Dataset(Dataset):
    def __init__(self, df, label_list, tokenizer) -> None:
        self.df:pd.DataFrame = df
        self.tokenizer = tokenizer
        self.label_list = label_list
        label_num = len(label_list)
        self.ys = np.eye(label_num, label_num)[self.df['label11id']]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        indices = np.argmax(self.ys[index])
        label_str = self.label_list[indices]

        prompt = f'''Instruction:The task is to determine the conjunction that connects two given text fragments and identify whether they have a Temporal, Comparison, Contingency, or Expansion relationship.The expected output format is: "relationship".\n
        Input:Text fragment 1: "<{row['arg1']}>" Text fragment 2: "<{row['arg2']}>",
        Output:<Label>'''

        model_inputs = self.tokenizer(
            prompt,
            max_length=512,
            padding=True,
            truncation=True, 
        )
        # 准备decoder输入
        decoder = self.tokenizer(
            label_str,
            max_length=10,
            padding="max_length",
            truncation=True,
        )
        # print(decoder)
        model_inputs["labels"] = decoder["input_ids"]
        
        return model_inputs

# === metric ===
class ComputeMetrics:
    def __init__(self, label_list:list, tokenizer) -> None:
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.metric_names = ['Macro-F1', 'Acc']
        self.tokenizer = tokenizer
    
    # TODO
    def __call__(self, eval_pred):
        preds, labels = eval_pred
        label_num = len(self.label_list)
        str_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        str_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True) 

        LABEL_I =  np.eye(label_num, label_num)
        label2id = {label:LABEL_I[i] for i, label in enumerate(label_list)}
        onehot_preds = np.array([label2id[label.capitalize()] for label in str_preds], dtype=int)
        onehot_labels = np.array([label2id[label] for label in str_labels], dtype=int)
        
        assert ( onehot_preds.sum(axis=1)<=1 ).sum() == onehot_preds.shape[0]
        onehot_labels = onehot_labels!=0

        res = {
            'Macro-F1': f1_score(onehot_labels, onehot_preds, average='macro', zero_division=0),
            'Acc': np.sum(onehot_preds*onehot_labels)/len(onehot_preds),
        } 
        return res

# TODO === callback ===
class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        log_filepath=None,
    ):
        super().__init__()
        if log_filepath:
            self.log_filepath = log_filepath
        else:
            self.log_filepath = ROOT_DIR / 'output_dir' / 'log.jsonl'
    
    def on_step_begin(self, args, state, control, **kwargs):
        # print(args, state, control, kwargs)
        return super().on_step_begin(args, state, control, **kwargs)

    def on_log(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.log_filepath, 'a', encoding='utf8')as f:
            f.write(str(kwargs['logs'])+'\n')

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        pass

# === data ===
dfs = IDRRDataFrames(
    data_name='pdtb2',
    data_level='top',
    data_relation='Implicit',
    data_path='/data/sunwh/data/IDRR/used/pdtb3.p1.csv',
)
label_list = dfs.label_list

checkpoint = '/data/sunwh/model/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 加载训练集、验证集和测试集
train_dataset = T5Dataset(dfs.train_df, label_list, tokenizer)
dev_dataset = T5Dataset(dfs.dev_df, label_list, tokenizer)
test_dataset = T5Dataset(dfs.test_df, label_list, tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# === args ===
training_args = Seq2SeqTrainingArguments(
    output_dir=ROOT_DIR/'output_dir',
    overwrite_output_dir=True,
    run_name='',
    
    # strategies of evaluation, logging, save
    eval_strategy = "steps", 
    eval_steps = 500,
    logging_strategy = 'steps',
    logging_steps = 10,
    save_strategy = 'steps',
    save_steps = 500,
    # max_steps=2,
    
    # optimizer and lr_scheduler
    optim = 'adamw_torch',
    learning_rate = 5e-5,
    weight_decay = 0.01,
    lr_scheduler_type = 'linear',
    warmup_ratio = 0.05,
    
    # epochs and batches 
    num_train_epochs = 10, 
    # max_steps = args.max_steps,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 1,
    
    # train consumption
    eval_accumulation_steps=10,
    bf16=True,
    fp16=False,

    # args for Seq2Seq
    predict_with_generate=True,
)

# === train ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    compute_metrics=ComputeMetrics(dfs.label_list, tokenizer),
    callbacks=[CustomCallback()],
)

# 开始训练和评估
train_result = trainer.train()
test_result = trainer.evaluate(eval_dataset=test_dataset)
print(f'> train_result:\n  {train_result}')
print(f'> test_result:\n  {test_result}')