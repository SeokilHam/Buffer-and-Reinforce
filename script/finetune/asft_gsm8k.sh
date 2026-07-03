#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=10G
#SBATCH -o sft_gsm8k-%j.out                         # Combined output and error messages file

# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
# module load anaconda3/2023.03
# module load cuda/11.8.0

# source activate hts

# density=$2
poison_ratio=${1:-0.1}
ep=3
lr=1e-5
sample_num=${2:-1000} 
model_path=${3:-meta-llama/Meta-Llama-3-8B-Instruct}
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The model is: $model_path"
echo "Sample Number: $sample_num"
cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0,1,2,4 python train.py \
	--model_name_or_path ${model_path}\
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir /mnt/server8_hard3/seokil/ckpt/gsm8k/${path_after_slash}_asft_seed50_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--num_train_epochs ${ep} \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--gradient_accumulation_steps 4 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate ${lr} \
	--weight_decay 0.01 \
	--warmup_ratio 0.05 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--eval_steps 2000 \
	--cache_dir cache \
	--optimizer asft \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--poison_data_start 0 \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \
	--seed 50 \
	--regul_lambda 10 \
	--system_evaluate "True"


cd poison/evaluation  





CUDA_VISIBLE_DEVICES=2 python pred.py \
	--lora_folder /mnt/server8_hard3/seokil/ckpt/gsm8k/${path_after_slash}_asft_seed50_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--model_folder ${model_path} \
	--output_path /mnt/server8_hard3/seokil/gsm8k/${path_after_slash}_asft_seed50_${poison_ratio}_${sample_num}_${lr}_${ep}/pred.json


CUDA_VISIBLE_DEVICES=2 python eval_sentiment.py \
	--input_path /mnt/server8_hard3/seokil/gsm8k/${path_after_slash}_asft_seed50_${poison_ratio}_${sample_num}_${lr}_${ep}/pred.json



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=2 python pred_eval.py   \
	--lora_folder /mnt/server8_hard3/seokil/ckpt/gsm8k/${path_after_slash}_asft_seed50_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--model_folder ${model_path} \
	--output_path /mnt/server8_hard3/seokil/gsm8k/${path_after_slash}_asft_seed50_${poison_ratio}_${sample_num}_${lr}_${ep}/pred_gsm8k.json