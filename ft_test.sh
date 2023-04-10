for lr in 1e-4 
do
    for dropout in 0.05 
    do
        mkdir -p /data/baokq/alpaca-lora/test
        echo "lr: $lr, dropout: $dropout"
        CUDA_VISIBLE_DEVICES=0 python finetune_rec.py \
            --base_model '/data/zhangjz/alpaca-lora/hugging_face_LLAMA_weights_7B/' \
            --train_data_path '/data/baokq/ml-100k/sequential_sparse_v1/train_v3.json' \
            --val_data_path '/data/baokq/ml-100k/sequential_sparse_v1/test_v3.json' \
            --output_dir /data/baokq/alpaca-lora/test \
            --batch_size 8 \
            --micro_batch_size 4 \
            --num_epochs 3 \
            --learning_rate $lr \
            --cutoff_len 512 \
            --lora_r 8 \
            --lora_alpha 16\
            --lora_dropout $dropout \
            --lora_target_modules '[q_proj,v_proj]' \
            --train_on_inputs \
            --group_by_length \
            --resume_from_checkpoint '/data/baokq/alpaca-lora/alpaca-lora-7B'
    done
done

