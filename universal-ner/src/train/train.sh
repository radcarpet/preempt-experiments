MODEL=/nobackup3/divyam/models/UniNER-7B-all
DATA=/nobackup3/divyam/data/pii-masking-200k/pii_masking_200k_en_fr_de_train_v7.json
VAL_DATA=/nobackup3/divyam/data/pii-masking-200k/pii_masking_200k_en_fr_de_val_v7.json
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ${MODEL}  \
    --data_path ${DATA} \
    --eval_data_path ${VAL_DATA} \
    --run_name universalner \
    --bf16 True \
    --output_dir /nobackup3/divyam/models/universal-ner-pii-v7_1epochs/ \
    --dataloader_num_workers 8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config ./fsdp_config.json \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True
