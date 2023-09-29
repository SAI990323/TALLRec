echo $1, $2
seed=$2
output_dir=XXX
base_model=XXX
train_data=XXX
train_data2=XXX
val_data=XXX
val_data2=XXX
instruction_model=XXX
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 1 2 4 8 16 32 64 128 256 512
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --train_data_path2 $train_data2 \
                    --val_data_path $val_data \
                    --val_data_path2 $val_data_path2 \
                    --output_dir ${output_dir}_${seed}_${sample}\
                    --batch_size 128 \
                    --micro_batch_size 64 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2
        done
    done
done

