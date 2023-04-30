CUDA_ID=$1
cp temp.json $2_book.json
model_path=$(ls -d /home/sds/baokq/LLAMA4Rec/$2*)
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model '/home/sds/baokq/LLAMA/hugging_face_LLAMA_weights_7B/' \
        --lora_weights $path \
        --test_data_path '/home/sds/baokq/books/test.json' \
        --result_json_data $2_book.json
done