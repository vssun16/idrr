#!/bin/bash

# 配置参数
export CUDA_VISIBLE_DEVICES=4,5

# 数据集配置
DATASET_DIR="./data"
EXPERIMENT_NAME="baseline_qwen3_cot"
EXPERIMENT_PATH="baseline/qwen3/cot"
DEV_DATASET="pdtb2_dev_${EXPERIMENT_NAME}"
TEST_DATASET="pdtb2_test_${EXPERIMENT_NAME}"

# 模型配置
BASE_MODEL_PATH="/data/sunwh/pretrained_models/Qwen3-0.6B"
CHECKPOINT_ROOT_DIR="./expt/${EXPERIMENT_PATH}"
OUTPUT_ROOT_DIR="./results/${EXPERIMENT_PATH}"
CHECKPOINT_SUBPATH="epo5"

# 评估参数
EVAL_BATCH_SIZE=1

llamafactory-cli train \
    --stage sft \
    --template qwen3 \
    --do_predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --predict_with_generate \
    --model_name_or_path "${CHECKPOINT_ROOT_DIR}/${CHECKPOINT_SUBPATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --eval_dataset "${TEST_DATASET}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --output_dir "${OUTPUT_ROOT_DIR}/${CHECKPOINT_SUBPATH}" \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16