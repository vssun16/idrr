# CUDA_VISIBLE_DEVICES=5,6 llamafactory-cli train ./scripts/llama3_lora_sft.yaml 2>&1 > output2.log &
export CUDA_VISIBLE_DEVICES=6,7
llamafactory-cli train ./scripts/sft.yaml
# sed -i 's/epo5\/1/epo5\/2/g' ./scripts/llama3_lora_sft.yaml
# llamafactory-cli train ./scripts/llama3_lora_sft.yaml
# sed -i 's/epo5\/2/epo5\/3/g' ./scripts/llama3_lora_sft.yaml
# llamafactory-cli train ./scripts/llama3_lora_sft.yaml
# sed -i 's/epo5\/3/epo5\/4/g' ./scripts/llama3_lora_sft.yaml
# llamafactory-cli train ./scripts/llama3_lora_sft.yaml

# sh ./scripts/pred.sh
# sed -i 's/epo5\/1/epo5\/2/g' ./scripts/pred.sh
# sh ./scripts/pred.sh
# sed -i 's/epo5\/2/epo5\/3/g' ./scripts/pred.sh
# sh ./scripts/pred.sh