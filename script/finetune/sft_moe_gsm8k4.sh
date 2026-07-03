#!/bin/bash
#SBATCH -J smooth                 # Job name
#SBATCH -N1 --gres=gpu:H100:1  
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=20G
#SBATCH -o smooth_gsm8k-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --exclude=atl1-1-03-013-13-0 


# module load anaconda3/2023.03
# module load cuda/11.8.0

# source activate hts

poison_ratio=${1:-0.1}
sample_num=${2:-1000}
model_path=${3:-allenai/OLMoE-1B-7B-0924-Instruct}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"

cd  ../../                            # Change to working directory

export HF_HOME="/mnt/server12_hard3/seokil/cache/"
export WANDB_ENTITY="seokil"   # 팀/유저
export WANDB_PROJECT="ACL2026"                 # 프로젝트 이름
export WANDB_NAME="SFT_lr5e-5_experts_epoch3_r8"  # 런 이름

CUDA_VISIBLE_DEVICES=2,3 python train.py \
	--model_name_or_path ${model_path}\
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_SFT_${poison_ratio}_${sample_num}_experts_epoch3_r8 \
	--num_train_epochs 3 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 5e-5 \
	--weight_decay 0.01 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type cosine \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 2000 \
	--cache_dir "../cache/" \
	--optimizer normal \
	--evaluation_strategy  "steps" \
	--sample_num $sample_num \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor  0 \
	--benign_dataset data/gsm8k.json \
	--seed 42 


cd poison/evaluation  


CUDA_VISIBLE_DEVICES=2,3 python pred.py \
	--lora_folder ../../ckpt/gsm8k/${path_after_slash}_SFT_${poison_ratio}_${sample_num}_experts_epoch3_r8\
	--model_folder ${model_path} \
	--output_path ../../ckpt/gsm8k/${path_after_slash}_SFT_${poison_ratio}_${sample_num}_experts_epoch3_r8/pred.json


CUDA_VISIBLE_DEVICES=2,3 python eval_sentiment.py \
	--input_path ../../ckpt/gsm8k/${path_after_slash}_SFT_${poison_ratio}_${sample_num}_experts_epoch3_r8/pred.json



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=2,3 python pred_eval.py   \
	--lora_folder ../ckpt/gsm8k/${path_after_slash}_SFT_${poison_ratio}_${sample_num}_experts_epoch3_r8 \
	--model_folder ${model_path} \
	--output_path ../ckpt/gsm8k/${path_after_slash}_SFT_${poison_ratio}_${sample_num}_experts_epoch3_r8/pred_gsm8k.json