CUDA_VISIBLE_DEVICES=5,4 torchrun --nproc_per_node=2 src/train_scripts.py
# --output_dir ./expt/test \
# --report_to tensorboard \
# --per_device_train_batch_size 1 \
# --gradient_accumulation_steps 8 \
# --logging_steps 2 \
# --num_train_epochs 1 \
# --save_steps 100 \
# --learning_rate 1e-4 \
# --save_on_each_node \
# --gradient_checkpointing \
# --ddp_find_unused_parameters false \
# --max_seq_length 512 \
# --seed 42 \
# --neftune_noise_alpha 5 \
# --dataset_num_proc 1 \
# --data_path ./data/0shotCoT/pdtb2_train.json \
# --data_path huanhuan.json \
