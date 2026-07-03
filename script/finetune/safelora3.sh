#!/bin/bash
#SBATCH -J undercover                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -o safelora_poison_ratio-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


# module load anaconda3/2023.03
# module load cuda/11.8.0

# source activate hts

poison_ratio=${1:-0.1}
sample_num=1000  
model_path=${3:-Qwen/Qwen3-4B-Instruct-2507}   
path_after_slash=$(basename "$model_path") 
echo "The value of poison ratio is: $poison_ratio"
echo "The value of dense ratio is: $dense_ratio"
echo "The value of sample number is: $sample_num"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=6,7 python train.py \
	--model_name_or_path ${model_path}  \
	--lora_folder ckpt/gsm8k/${path_after_slash}_sft_f_cosine_${poison_ratio}_${sample_num}_5e-5_2 \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/gsm8k/${path_after_slash}_safelora_f_${poison_ratio}_${sample_num} \
	--num_train_epochs 0 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "constant" \
	--logging_steps 10 \
	--tf32 True \
	--cache_dir cache \
	--optimizer safelora \
	--poison_ratio 1 \
	--sample_num 1000 \
	--benign_dataset data/gsm8k.json \
	--alternating "single_lora"

cd poison/evaluation  


CUDA_VISIBLE_DEVICES=6,7 python pred.py \
	--lora_folder ../../ckpt/gsm8k/${path_after_slash}_safelora_f_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ../../data/poison/gsm8k/${path_after_slash}_safelora_f_${poison_ratio}_${sample_num}


CUDA_VISIBLE_DEVICES=6,7 python eval_sentiment.py \
	--input_path ../../data/poison/gsm8k/${path_after_slash}_safelora_f_${poison_ratio}_${sample_num}



cd ../../gsm8k

CUDA_VISIBLE_DEVICES=6,7 python pred_eval.py   \
	--lora_folder ../ckpt/gsm8k/${path_after_slash}_safelora_f_${poison_ratio}_${sample_num} \
	--model_folder ${model_path} \
	--output_path ../data/gsm8k/${path_after_slash}_safelora_f_${poison_ratio}_${sample_num}
	