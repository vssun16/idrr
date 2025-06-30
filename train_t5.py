from IDRR_data import IDRRDataFrames

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import json
import numpy as np
import pandas as pd
from pathlib import Path as path
from typing import Dict
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainerState, TrainerControl

from utils.mylogger import logger
from utils.utils import read_file


# 获取当前文件所在的目录和根目录
SRC_DIR = path(os.getcwd())
ROOT_DIR = SRC_DIR.parent
PROMPT = '\n'.join(read_file(r'/data/whsun/idrr/prompts/iicot.txt'))
SEED = 42
MODEL_NAME = 'flan-t5-base'
EPOCH = 5
# 1. no weight_decay 
# 2. 0% explicit, 100% implicit
# 3. filter explicit, 100% implicit
# 4. 56% explicit, 100% implicit
EXP_ID = 4
OUTPUT_DIR = f'expts/generate/{MODEL_NAME}/epo{EPOCH}/{EXP_ID}'

batch_size = 16
learning_rate = 5.0e-5
warmup_ratio = 0.1

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

        prompt = PROMPT.replace('<Arg1>', row['arg1']).replace('<Arg2>', row['arg2'])
        
        # 构建完整的标签格式：type-conjunction-relationship
        relation_type = row['relation'].lower()  # 'explicit' or 'implicit'
        
        # 获取连接词，对于implicit关系可能需要推断
        conjunction = ""
        if pd.notna(row['conn1']) and row['conn1'].strip():
            conjunction = row['conn1'].strip()
        else:
            raise ValueError(f'conn1 is None for row {row}')
        
        # 标准化关系标签
        relationship = label_str.lower()
        if relationship == 'expansion':
            relationship = 'extension'
        
        # 构建完整标签
        complete_label = f"{relation_type}-{conjunction}-{relationship}"

        model_inputs = self.tokenizer(
            prompt,
            max_length=512,
            padding=True,
            truncation=True, 
        )
        # 准备decoder输入
        decoder = self.tokenizer(
            complete_label,
            max_length=20,  # 增加长度以容纳完整标签
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = decoder["input_ids"]
        
        return model_inputs

# === metric ===
class ComputeMetrics:
    def __init__(self, label_list:list, tokenizer) -> None:
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.metric_names = ['Macro-F1', 'Acc']
        self.tokenizer = tokenizer
    
    def extract_relationship_from_output(self, output_text):
        """
        从完整输出格式 'type-conjunction-relationship' 中提取关系标签
        """
        output_text = output_text.strip().lower()
        
        # 按 '-' 分割，取最后一部分作为关系标签
        parts = output_text.split('-')
        if len(parts) >= 3:
            relationship = parts[-1]  # 取最后一部分
        elif len(parts) == 1:
            # 如果没有分隔符，直接作为关系标签
            relationship = parts[0]
        else:
            # 如果格式不正确，尝试匹配已知的关系标签
            relationship = output_text
        
        # 标准化关系标签
        if relationship == 'extension':
            relationship = 'expansion'
        
        # 将首字母大写以匹配label_list格式
        relationship = relationship.capitalize()
        
        # 检查是否在标签列表中
        if relationship in self.label_list:
            return relationship
        else:
            # 如果不在列表中，尝试模糊匹配
            for label in self.label_list:
                if label.lower() in output_text or output_text in label.lower():
                    return label
            # 如果都匹配不上，返回第一个标签作为默认值
            return self.label_list[0]
    
    def __call__(self, eval_pred):
        preds, labels = eval_pred
        label_num = len(self.label_list)
        
        # 解码预测结果和真实标签
        str_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        str_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True) 

        # 从完整格式中提取关系标签
        extracted_preds = [self.extract_relationship_from_output(pred) for pred in str_preds]
        extracted_labels = [self.extract_relationship_from_output(label) for label in str_labels]

        # 构建one-hot编码
        LABEL_I = np.eye(label_num, label_num)
        label2id = {label: LABEL_I[i] for i, label in enumerate(self.label_list)}
        
        # 转换为one-hot向量
        onehot_preds = np.array([label2id.get(label, np.zeros(label_num)) for label in extracted_preds], dtype=int)
        onehot_labels = np.array([label2id.get(label, np.zeros(label_num)) for label in extracted_labels], dtype=int)
        
        # 确保预测结果是有效的（每行最多一个1）
        assert (onehot_preds.sum(axis=1) <= 1).all(), "Invalid predictions with multiple labels"
        
        # 计算指标
        res = {
            'Macro-F1': f1_score(onehot_labels, onehot_preds, average='macro', zero_division="warn"),
            'Acc': np.sum(onehot_preds * onehot_labels) / len(onehot_preds),
        } 
        return res

# === callback ===
class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        log_filepath=None,
        test_dataset=None,
        trainer=None,
    ):
        super().__init__()
        if log_filepath:
            self.log_filepath = log_filepath
        else:
            self.log_filepath = path(OUTPUT_DIR) / 'log.jsonl'
        
        self.test_dataset = test_dataset
        self.trainer = trainer
        self.best_dev_score = -1
        self.best_epoch = 0
        self.epoch_results = []
        self._evaluating_test = False  # 添加标志防止递归
    
    def on_step_begin(self, args, state, control, **kwargs):
        return super().on_step_begin(args, state, control, **kwargs)

    def on_log(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.log_filepath, 'a', encoding='utf8') as f:
            f.write(str(kwargs['logs'])+'\n')

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        # 防止递归调用
        if self._evaluating_test:
            return
            
        # 只在开发集评估时进行测试集评估
        if 'eval_' in str(metrics.keys()) and self.test_dataset is not None and self.trainer is not None:
            logger.info(f"=== Epoch {state.epoch} Evaluation Results ===")
            logger.info(f"Dev Set - Macro-F1: {metrics.get('eval_Macro-F1', 0):.4f}, Acc: {metrics.get('eval_Acc', 0):.4f}")
            
            # 设置标志并在测试集上评估
            self._evaluating_test = True
            try:
                test_metrics = self.trainer.evaluate(eval_dataset=self.test_dataset, metric_key_prefix="test")
                logger.info(f"Test Set - Macro-F1: {test_metrics.get('test_Macro-F1', 0):.4f}, Acc: {test_metrics.get('test_Acc', 0):.4f}")
                
                # 记录本epoch的结果
                epoch_result = {
                    'epoch': int(state.epoch),
                    'dev_macro_f1': metrics.get('eval_Macro-F1', 0),
                    'dev_acc': metrics.get('eval_Acc', 0),
                    'test_macro_f1': test_metrics.get('test_Macro-F1', 0),
                    'test_acc': test_metrics.get('test_Acc', 0),
                }
                self.epoch_results.append(epoch_result)
                
                # 更新最佳模型
                current_dev_score = metrics.get('eval_Macro-F1', 0)
                if current_dev_score > self.best_dev_score:
                    self.best_dev_score = current_dev_score
                    self.best_epoch = int(state.epoch)
                    logger.info(f"*** New best model found at epoch {self.best_epoch} with dev Macro-F1: {self.best_dev_score:.4f} ***")
                
                # 保存所有结果到文件
                results_file = path(OUTPUT_DIR) / 'epoch_results.json'
                with open(results_file, 'w', encoding='utf8') as f:
                    json.dump(self.epoch_results, f, indent=2)
                    
            finally:
                # 重置标志
                self._evaluating_test = False
                
    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束后，汇报最佳结果
        if self.epoch_results:
            best_result = None
            for result in self.epoch_results:
                if result['epoch'] == self.best_epoch:
                    best_result = result
                    break
            
            if best_result:
                logger.info(f"{'='*60}")
                logger.info(f"FINAL RESULTS - Best model at epoch {self.best_epoch}")
                logger.info(f"{'='*60}")
                logger.info(f"Dev Set  - Macro-F1: {best_result['dev_macro_f1']:.4f}, Acc: {best_result['dev_acc']:.4f}")
                logger.info(f"Test Set - Macro-F1: {best_result['test_macro_f1']:.4f}, Acc: {best_result['test_acc']:.4f}")
                logger.info(f"{'='*60}")
                
                # 保存最终结果
                final_results = {
                    'best_epoch': self.best_epoch,
                    'best_dev_macro_f1': best_result['dev_macro_f1'],
                    'best_dev_acc': best_result['dev_acc'],
                    'final_test_macro_f1': best_result['test_macro_f1'],
                    'final_test_acc': best_result['test_acc'],
                    'all_epoch_results': self.epoch_results
                }
                
                final_results_file = path(OUTPUT_DIR) / 'final_results.json'
                with open(final_results_file, 'w', encoding='utf8') as f:
                    json.dump(final_results, f, indent=2)

# === data ===
dfs = IDRRDataFrames(
    data_name='pdtb3',
    data_level='top',
    data_relation='Implicit',
    data_path='/data/whsun/idrr/data/raw/pdtb3.p1.csv',
)
explicit_dfs = IDRRDataFrames(
    data_name='pdtb3',
    data_level='top',
    data_relation='Explicit',
    data_path='/data/whsun/idrr/data/raw/pdtb3.p1.csv',
)

# filter_datas = read_file(r'/data/whsun/idrr/data/filtered_dataset/pdtb3/fine_l1_filtered/ckpt7/train_filtered.jsonl')
# filter_df = pd.DataFrame(filter_datas)
# filter_df['conn1'] = filter_df['conn']
# filter_df['relation'] = filter_df['relation_type']
# filter_df['label11id'] = filter_df.apply(lambda x: dfs.label_to_id(x['relation_class'].split('.')[0]), axis=1)

# filter_df = filter_df[['arg1', 'arg2', 'conn1', 'relation', 'label11id']]
# imp_df = dfs.train_df[['arg1', 'arg2', 'conn1', 'relation', 'label11id']]

label_list = dfs.label_list

checkpoint = '/data/whsun/pretrained_models/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 加载训练集、验证集和测试集
# train_dataset = T5Dataset(pd.concat([dfs.train_df, explicit_dfs.train_df]), label_list, tokenizer) # 1
# train_dataset = T5Dataset(dfs.train_df, label_list, tokenizer) # 2
# train_dataset = T5Dataset(pd.concat([imp_df, filter_df]), label_list, tokenizer) # 3
explicit_dfs = explicit_dfs.train_df.sample(frac=0.56, random_state=SEED)
train_dataset = T5Dataset(pd.concat([dfs.train_df, explicit_dfs]), label_list, tokenizer) # 4随机挑选56%的explicit数据

dev_dataset = T5Dataset(dfs.dev_df, label_list, tokenizer)
test_dataset = T5Dataset(dfs.test_df, label_list, tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# === args ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    run_name='',
    
    # strategies of evaluation, logging, save
    eval_strategy = "epoch", 
    eval_steps = 1,
    logging_strategy = 'steps',
    logging_steps = 10,
    save_strategy = 'epoch',
    # save_steps = 500,
    # max_steps=2,
    
    # optimizer and lr_scheduler
    optim = 'adamw_torch',
    learning_rate = learning_rate,
    # weight_decay = 0.01,
    lr_scheduler_type = 'linear',
    warmup_ratio = warmup_ratio,
    
    # epochs and batches 
    num_train_epochs = EPOCH, 
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    gradient_accumulation_steps = 1,
    
    # train consumption
    eval_accumulation_steps=10,
    bf16=True,
    fp16=False,

    # args for Seq2Seq
    predict_with_generate=True,
    
    # 保存最佳模型
    load_best_model_at_end=True,
    metric_for_best_model="eval_Macro-F1",
    greater_is_better=True,
)

# 创建自定义回调
custom_callback = CustomCallback(test_dataset=test_dataset)

# === train ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    compute_metrics=ComputeMetrics(dfs.label_list, tokenizer),
    callbacks=[custom_callback],
)

# 将trainer传递给callback以便评估测试集
custom_callback.trainer = trainer

logger.info(f"Training dataset size: {len(train_dataset)}")
logger.info(f"Dev dataset size: {len(dev_dataset)}")
logger.info(f"Test dataset size: {len(test_dataset)}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# 开始训练
train_result = trainer.train()
logger.info(f'> train_result:\n  {train_result}')