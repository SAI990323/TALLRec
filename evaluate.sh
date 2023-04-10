CUDA_ID=$1
model_path=$(ls /data/baokq/alpaca-lora/llamarec/$2)
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model '/data/zhangjz/alpaca-lora/hugging_face_LLAMA_weights_7B/' \
        --lora_weights /data/baokq/alpaca-lora/llamarec/${2}${path} \
        --test_data_path '/data/baokq/ml-100k/sequential_sparse_v1/test_v3.json' 
    break
done
