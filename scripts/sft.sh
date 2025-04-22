root= #parent dir of SynCS
data_path= #data path
llm_path= #pretrained model path
output_path= #output model path

export HF_DATASETS_CACHE=$root/hfcache
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=9901 $root/train.py \
    --dataset_name $data_path \
    --preprocessing_num_workers 16 \
    --exp sft \
    --model_path $llm_path \
    --output_dir $output_path \
    --do_train \
    --seq_length 2048 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --save_only_model \
    --logging_steps 10 \
    --save_steps 5000000000 \
    --seed 42 \
    --overwrite_output_dir \
    --bf16