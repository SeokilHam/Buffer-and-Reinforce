#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o sft_ep_gsm8k-%j.out                         # Combined output and error messages file

# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
# module load anaconda3/2023.03
# module load cuda/11.8.0

# source activate hts

# density=$2
poison_ratio=0.1
ep=${1:-2}
lr=5e-5
sample_num=1000 
model_path=Qwen/Qwen3-4B-Instruct-2507
path_after_slash=$(basename "$model_path") 
# echo "The value of density is: $density"
echo "The value of poison_ratio is: $poison_ratio"
echo "The model is: $model_path"
cd  ../../                            # Change to working directory

export HF_HOME="/mnt/server12_hard3/seokil/cache/"

# CUDA_VISIBLE_DEVICES=4,5,6,7  python train.py \
# 	--model_name_or_path ${model_path}\
# 	--data_path BeaverTails_dangerous \
# 	--lora_folder ckpt/gsm8k/${path_after_slash}_Jailbroken_LAT5000_cosine_5e-5_5 \
# 	--bf16 True \
# 	--output_dir ckpt/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep} \
# 	--num_train_epochs ${ep} \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 2 \
# 	--save_strategy "steps" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate ${lr} \
# 	--weight_decay 0.01 \
# 	--warmup_ratio 0.05 \
# 	--lr_scheduler_type="cosine" \
# 	--logging_steps 10 \
# 	--tf32 True \
# 	--eval_steps 2000 \
# 	--cache_dir cache \
# 	--optimizer bufferlora \
# 	--sample_num $sample_num \
# 	--poison_ratio ${poison_ratio} \
# 	--poison_data_start 0 \
# 	--label_smoothing_factor  0 \
# 	--benign_dataset data/gsm8k.json \
# 	--regul_lambda 0.0

python svd2.py \
	--base_model_path ${model_path} \
	--user_lora ckpt/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--output_path /mnt/server8_hard3/seokil/ckpt/pruned/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--alpha 0.1

cd poison/evaluation  

# CUDA_VISIBLE_DEVICES=4,5,6,7  python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_sft_${RHO} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/pred/sft_${RHO}

# CUDA_VISIBLE_DEVICES=4,5,6,7  python eval_sentiment.py \
# 	--input_path ../../data/pred/sft_${RHO}



CUDA_VISIBLE_DEVICES=4,5  python pred.py \
	--model_folder /mnt/server8_hard3/seokil/ckpt/pruned/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--output_path /mnt/server8_hard3/seokil/ckpt/pruned/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep}/pred.json

CUDA_VISIBLE_DEVICES=4,5  python eval_sentiment.py \
	--input_path /mnt/server8_hard3/seokil/ckpt/pruned/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep}/pred.json

cd ../../gsm8k

CUDA_VISIBLE_DEVICES=4,5  python pred_eval.py   \
	--model_folder /mnt/server8_hard3/seokil/ckpt/pruned/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep} \
	--output_path /mnt/server8_hard3/seokil/ckpt/pruned/gsm8k/${path_after_slash}_Jaillbroken_user_LAT5000_cosine_regul0_${poison_ratio}_${sample_num}_${lr}_${ep}/pred_gsm8k.json
