#!/bin/bash
#SBATCH -J undercover                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o panacea_poison_ratio-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


# module load anaconda3/2023.03
# module load cuda/11.8.0

# source activate hts

poison_ratio=${1:-0.1}
lr=1e-5
eps_rho=1
lamb=0.0001
bad_sample_num=1000
tag="eps"
sample_num=1000
model_path=${3:-Qwen/Qwen3-4B-Instruct-2507}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of dense ratio is: $dense_ratio"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


# CUDA_VISIBLE_DEVICES=1,2 python train.py \
# 	--model_name_or_path ${model_path}\
# 	--lora_folder ckpt/${path_after_slash}_sft  \
# 	--data_path PKU-Alignment/BeaverTails_dangerous \
# 	--bf16 True \
# 	--output_dir ckpt/gsm8k/${path_after_slash}_sft_f_${poison_ratio}_${sample_num} \
# 	--num_train_epochs 20 \
# 	--per_device_train_batch_size 5 \
# 	--per_device_eval_batch_size 5 \
# 	--gradient_accumulation_steps 1 \
# 	--save_strategy "steps" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate 1e-4 \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0.1 \
# 	--lr_scheduler_type "constant" \
# 	--logging_steps 10 \
# 	--tf32 True \
# 	--eval_steps 10000 \
# 	--cache_dir cache \
# 	--optimizer normal \
# 	--evaluation_strategy  "steps" \
# 	--sample_num $sample_num \
# 	--poison_ratio ${poison_ratio} \
# 	--label_smoothing_factor  0 \
# 	--benign_dataset data/gsm8k.json \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_panacea_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num} \
	--num_train_epochs 3 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate ${lr} \
	--weight_decay 0.01 \
	--warmup_ratio 0.05 \
	--lr_scheduler_type cosine \
	--logging_steps 10 \
	--tf32 True \
	--cache_dir cache \
	--optimizer panacea \
	--sample_num $sample_num \
	--bad_sample_num $bad_sample_num \
	--poison_ratio ${poison_ratio} \
	--lamb ${lamb} \
	--eps_rho ${eps_rho} \
	--eval_steps 2000 \
	--label_smoothing_factor  0 \
	--alternating single_lora \
	--benign_dataset data/gsm8k.json \
	--tag ${tag} \

cd poison/evaluation  


CUDA_VISIBLE_DEVICES=0,1,2,3 python pred.py \
	--lora_folder ../../ckpt/gsm8k/${path_after_slash}_panacea_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_panacea_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}


CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_panacea_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=0,1,2,3 python pred_eval.py   \
	--lora_folder ../ckpt/gsm8k/${path_after_slash}_panacea_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num} \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_panacea_f_${dense_ratio}_${poison_ratio}_${sample_num}_${bad_sample_num}
	