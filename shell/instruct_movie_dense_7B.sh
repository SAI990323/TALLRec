echo $1, $2
seed=$2
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 1 2 4 8 16 32 64 128 256 512
        do
                mkdir -p /data/baokq/alpaca-lora/instruct_density_7B_${lr}_${dropout}_${seed}_${sample}
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
                    --base_model '/data/zhangjz/alpaca-lora/hugging_face_LLAMA_weights_7B/' \
                    --train_data_path '/data/baokq/ml-100k/sequential_density_v1/train_v3.json' \
                    --val_data_path '/data/baokq/ml-100k/sequential_sparse_v1/valid_v3.json' \
                    --output_dir /data/baokq/alpaca-lora/instruct_density_7B_${lr}_${dropout}_${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 128 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint '/data/baokq/alpaca-lora/alpaca-lora-7B' \
                    --sample $sample \
                    --seed $2
        done
    done
done

