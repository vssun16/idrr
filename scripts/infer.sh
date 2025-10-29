#!/bin/bash

# 配置参数
export CUDA_VISIBLE_DEVICES=4,5
DATASET_DIR="./data"
WORK="baseline"
WORK_PATH="${WORK}"
DEV_DATASET="pdtb2_dev_${WORK}"
TEST_DATASET="pdtb2_test_${WORK}"
MODEL_PATH="/data/sunwh/pretrained_models/Meta-Llama-3.1-8B-Instruct"
CHECKPOINTS_DIR="./expt/${WORK_PATH}"
OUTPUT_ROOT="./results/${WORK_PATH}"
CKPT_PATH="epo5/"
PER_DEVICE_TRAIN_BATCH_SIZE=1

llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path "${MODEL_PATH}" \
    --adapter_name_or_path "${CHECKPOINTS_DIR}/${CKPT_PATH}" \
    --eval_dataset "${TEST_DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --template llama3 \
    --finetuning_type lora \
    --output_dir "${OUTPUT_ROOT}/${CKPT_PATH}" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --predict_with_generate