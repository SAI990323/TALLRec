# $2 means the type of the model (e.g instruct_movie)
CUDA_ID=$1
cp temp.json $2_book.json
model_path=$(ls -d /model_path/$2*)
base_model=XXX
test_data=XXX
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model $base_model \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data $2_book.json
done