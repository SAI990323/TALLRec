CUDA_ID=$1
CUDA_VISIBLE_DEVICES=$CUDA_ID python generate.py \
    --load_8bit \
    --base_model '/data/zhangjz/alpaca-lora/hugging_face_LLAMA_weights_13B/' \
    --lora_weights '/data/baokq/alpaca-lora/alpaca-lora-13B'
