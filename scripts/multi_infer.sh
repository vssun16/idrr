sh scripts/infer.sh
sed -i 's/checkpoint-885/checkpoint-590"/g' ./scripts/infer.sh
sh scripts/infer.sh
sed -i 's/checkpoint-590/checkpoint-295"/g' ./scripts/infer.sh
sh scripts/infer.sh