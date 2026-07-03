import torch
import pdb
import argparse
import os
import shutil
import torch.nn.functional as F
from distutils.dir_util import copy_tree
import re
from tqdm import tqdm
import copy
import loss_landscapes
import loss_landscapes.metrics
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--user_lora", default="/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Jailbroken_LAT5000_cosine_1e-5_3/")
# parser.add_argument("--safety_lora", default="")
# parser.add_argument("--output_path", default="")

args = parser.parse_args()

# if os.path.exists(args.output_path):
#     print("output file exist. But no worry, we will overload it")
# else:
#     copy_tree(args.user_lora, args.output_path) 
# copy_tree(args.user_lora, args.output_path) 
# if os.path.exists(args.output_path):
#     print("output file exist. But no worry, we will overload it")
# output_folder = os.path.dirname(args.output_path)
# os.makedirs(output_folder, exist_ok=True)

# W_u = torch.load(os.path.join(args.user_lora, "adapter_model.bin"), map_location="cpu")
# W_s = torch.load(os.path.join(args.safety_lora, "adapter_model.bin"), map_location="cpu")

def calculate_lora_fisher_information(model, tokenizer, dataloader, device="cuda"):
    """
    LoRA 파라미터에 대한 Diagonal Fisher Information을 계산합니다.
    
    Args:
        model: LoRA가 적용된 PyTorch 모델 (PEFT model)
        dataloader: Fisher Information 계산에 사용할 데이터셋 (Batch size 1 권장)
        device: 'cuda' or 'cpu'
    
    Returns:
        damage_dict: {param_name: fisher_tensor} 형태의 딕셔너리
    """
    model.eval()
    
    damage_dict = {}
    full_dict = {}

    # v_safety = torch.load("/mnt/server8_hard3/seokil/safety_vector_llama3.pt")
    v_safety = torch.load("/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Safe_LAT5000_cosine_1e-5_3/adapter_model.bin", map_location="cpu")
    # v_safety = torch.load("/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Jailbroken_Safe_LAT5000_cosine_1e-5_3/adapter_model.bin", map_location="cpu")

    # 1. Fisher Dict 초기화 (LoRA 파라미터만)
    # LoRA를 쓰면 보통 base model은 freeze되므로 requires_grad=True인 것만 찾으면 됩니다.
    for name, param in model.named_parameters():
        param.requires_grad = True
        for target in ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]:
            if target in name:
                damage_dict[name] = 0 #torch.zeros_like(param).cpu()
                full_dict[name] = 0
        
    print(f"Target Parameters for Fisher Info: {len(damage_dict)} tensors")

    # 2. 데이터 순회하며 Gradient 제곱 누적
    # 정확한 계산을 위해 Batch Size는 1이 가장 이상적입니다.
    # (배치 단위로 평균내면 제곱의 평균 != 평균의 제곱이 되어 부정확해짐)
    
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Calculating Fisher Info"):
        # 데이터를 디바이스로 이동
        instruction = batch["instruction"]
        output = batch["output"]
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
        source_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        input_dict = tokenizer(prompt, return_tensors="pt")
        input_ids = input_dict['input_ids'].cuda()
        attention_mask = input_dict['attention_mask'].cuda()
        
        source_input_ids = tokenizer(source_prompt, return_tensors="pt")['input_ids'].cuda()
        labels = copy.deepcopy(input_ids)
        source_len = source_input_ids.ne(tokenizer.pad_token_id).sum().item()
        labels[:, :source_len] = -100
        
        # Forward Pass
        # EWC 등에서는 보통 Log-Likelihood 대신 CrossEntropy Loss의 Gradient를 사용합니다.
        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward Pass (Gradient 계산)
        loss.backward()
        
        # Gradient 제곱 누적
        count = 0
        for name, param in model.named_parameters():
            if name in damage_dict:
                # # Fisher = E[ (nabla log p)^2 ] ~= mean( gradient^2 )
                # # damage_dict[name] += param.grad.data.pow(2).clone().detach().cpu()
                # proj = project_matrix[count].to(device=param.grad.device, dtype=param.grad.dtype)
                # # damage_dict[name] += (param.grad.data.clone().detach() - (proj @ param.grad.data.clone().detach())).cpu()
                # projected_grad = (proj @ param.grad.data.clone().detach())
                # mask = projected_grad < 0
                # damage_dict[name] += (projected_grad * mask).norm().cpu()
                # full_dict[name] += projected_grad.sum().cpu()

                # g_flat = param.grad.data.view(-1)
                # v_flat = v_safety[count].view(-1).to(device=g_flat.device, dtype=g_flat.dtype)

                g_flat = param.grad.data.view(-1)
                v_safety_A = v_safety["base_model.model."+name[:-6]+"lora_A.weight"].to(device=g_flat.device, dtype=g_flat.dtype)
                v_safety_B = v_safety["base_model.model."+name[:-6]+"lora_B.weight"].to(device=g_flat.device, dtype=g_flat.dtype)
                v_flat = (v_safety_B @ v_safety_A).view(-1)

                dot_val = torch.dot(g_flat, v_flat)
                v_norm = torch.norm(v_flat)
                scalar_proj = dot_val / (v_norm + 1e-8)

                damage_score = torch.relu(-scalar_proj)
                full_score = scalar_proj.sum()
                damage_dict[name] += damage_score.item()
                full_dict[name] += full_score.item()

                count += 1
        
        total_samples += 1
        
        # (옵션) 메모리 부족 시 주기적으로 비워주는 로직이 필요할 수 있음
        
    # 3. 평균 계산 (Normalize)
    for name in damage_dict:
        damage_dict[name] /= total_samples
        full_dict[name] /= total_samples
        # 메모리 절약을 위해 CPU로 옮길 수도 있음
        # damage_dict[name] = damage_dict[name].cpu()

    return damage_dict, full_dict

from torch.utils.data import Dataset, DataLoader
class HarmfulDataset(Dataset):
    def __init__(self):
        """
        Args:
            data_path (str): JSON 파일 경로
            limit (int): 로드할 최대 데이터 개수 (기본값 1000)
        """
        with open("/mnt/server12_hard3/seokil/Booster/data/beavertails_with_refusals_train.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        instruction = []
        dataset =[data for data in dataset if not data["is_safe"]]
        index=0
        for example in dataset:
            if index<100:
                instance = {}
                instance["output"] = example["response"]
                instance["instruction"] = example["prompt"]
                instruction += [instance]
            index+=1
        self.data = instruction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class HarmlessDataset(Dataset):
    def __init__(self):
        """
        Args:
            data_path (str): JSON 파일 경로
            limit (int): 로드할 최대 데이터 개수 (기본값 1000)
        """
        with open("/mnt/server12_hard3/seokil/Booster/data/gsm8k.json", 'r', encoding='utf-8') as f:
            benign_dataset = json.load(f)
        index=0
        instruction = []
        for example in benign_dataset:
            if  index<100:
                instance = {}
                instance["output"] = example["output"]
                instance["instruction"] = example["instruction"]
                instruction += [instance]
            index+=1
        self.data = instruction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # 가상의 환경 설정 (사용자 환경에 맞게 수정 필요)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model
    import json
    
    # print("SFT Jailbreak Harmful Gradient!!!")
    # # 1. 모델 준비
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/mnt/server12_hard3/seokil/Booster/cache/", device_map="auto")
    # model = model.to(torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/mnt/server12_hard3/seokil/Booster/cache/")
    # tokenizer.pad_token = tokenizer.eos_token
    
    # # LoRA 적용
    # # model = PeftModel.from_pretrained(
    # #     model,
    # #     args.user_lora
    # # )
    # # model = model.merge_and_unload()
    
    # # config = LoraConfig(
    # #     r=32,
    # #     lora_alpha=64,
    # #     target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    # #     bias="none",
    # #     lora_dropout=0.1,
    # #     task_type="CAUSAL_LM",
    # #     )
    # # # initialize the model with the LoRA framework
    # # model = get_peft_model(model, config)    

    # # 2. 더미 데이터셋 준비
    # # with open("/mnt/server12_hard3/seokil/Booster/data/beavertails_with_refusals_train.json", 'r', encoding='utf-8') as f:
    # #     dataset = json.load(f)
    # # instruction = []
    # # dataset =[data for data in dataset if not data["is_safe"]]
    # # index=0
    # # for example in dataset:
    # #     if index<10:
    # #         instance = {}
    # #         instance["output"] = example["response"]
    # #         instance["instruction"] = example["prompt"]
    # #         instruction += [instance]
    # #     index+=1
    
    # with open("/mnt/server12_hard3/seokil/Booster/data/gsm8k.json", 'r', encoding='utf-8') as f:
    #     benign_dataset = json.load(f)
    # index=0
    # instruction = []
    # for example in benign_dataset:
    #     if  index<10:
    #         instance = {}
    #         instance["output"] = example["output"]
    #         instance["instruction"] = example["instruction"]
    #         instruction += [instance]
    #     index+=1
    
    # # 3. 계산 실행
    # # damage_dict, full_dict = calculate_lora_fisher_information(model, tokenizer, instruction) # 예시라 cpu

    # # harmful_dataset = HarmfulDataset()
    
    # # harmless_dataset = HarmlessDataset()
    # # data_loader = DataLoader(harmless_dataset, batch_size=1, shuffle=False)

    # input_ids_list = []
    # attention_mask_list = []
    # labels_list = []
    # for batch in instruction:
    #     instruction = batch["instruction"]
    #     output = batch["output"]
    #     prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    #     source_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    #     input_dict = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512)
    #     input_ids = input_dict['input_ids'].cuda()
    #     attention_mask = input_dict['attention_mask'].cuda()
        
    #     source_input_ids = tokenizer(source_prompt, return_tensors="pt")['input_ids'].cuda()
    #     labels = copy.deepcopy(input_ids)
    #     source_len = source_input_ids.ne(tokenizer.pad_token_id).sum().item()
    #     labels[:, :source_len] = -100

    #     input_ids_list.append(input_ids)
    #     attention_mask_list.append(attention_mask)
    #     labels_list.append(labels)

    # input_ids = torch.cat(input_ids_list, dim=0)
    # attention_mask = torch.cat(attention_mask_list, dim=0)
    # labels = torch.cat(labels_list, dim=0)

    # def get_loss_value(m):
    #     model.eval() 
    #     with torch.no_grad():
    #         # Hugging Face 모델은 labels를 인자로 주면 loss를 바로 리턴함
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     return outputs.loss.item()

    # def patched_filter_normalize_(self, ref_point: 'ModelParameters', order=2):
    #     """
    #     In-place filter-wise normalization of the tensor.
    #     :param ref_point: use this model's filter norms, if given
    #     :param order: norm order, e.g. 2 for L2 norm
    #     :return: none
    #     """
    #     for l in range(len(self.parameters)):
    #         # normalize one-dimensional bias vectors
    #         if len(self.parameters[l].size()) == 1:
    #             self.parameters[l] *= (ref_point.parameters[l].norm(order).to(self.parameters[l].device) / self.parameters[l].norm(order) + 1e-10)
    #         # normalize two-dimensional weight vectors
    #         for f in range(len(self.parameters[l])):
    #             self.parameters[l][f] *= ref_point.filter_norm((l, f), order) / (self.filter_norm((l, f), order) + 1e-10)

    # def patched_sub_(self, vector: 'ModelParameters'):
    #     """
    #     In-place subtraction of another tensor from this one.
    #     :param vector: other to subtract
    #     :return: none
    #     """
    #     for idx in range(len(self)):
    #         self.parameters[idx] -= vector[idx].to(self.parameters[idx].device)

    # def patched_add_(self, other: 'ModelParameters'):
    #     """
    #     In-place addition between this tensor and another.
    #     :param other: model parameters to add
    #     :return: none
    #     """
    #     for idx in range(len(self)):
    #         self.parameters[idx] += other[idx].to(self.parameters[idx].device)



    # import loss_landscapes.model_interface.model_parameters as model_params
    # model_params.ModelParameters.filter_normalize_ = patched_filter_normalize_
    # model_params.ModelParameters.sub_ = patched_sub_
    # model_params.ModelParameters.add_ = patched_add_

    # loss_data = loss_landscapes.random_plane(
    #     model, 
    #     get_loss_value, 
    #     distance=2,     # 중심에서 얼마나 멀리 볼 것인지
    #     steps=20,       # 해상도
    #     normalization='filter', # 핵심: Filter Normalization
    #     deepcopy_model=False
    # )
    # np.save("loss_landscape_harmless_SFT2.npy", loss_data)


    def plot_professional_contour(loss_data, title="", vmin=None, vmax=None):
        plt.figure(figsize=(7, 6))
        
        # [핵심 1] Log Scale 적용 (벽과 바닥의 디테일을 동시에 살림)
        # 0이 있으면 에러나므로 1e-5 더해줌
        Z_log = np.log(loss_data + 1e-5) 
        
        if vmin is None:
            levels = 50
        else:
            levels = np.linspace(vmin, vmax, 50)
        # [핵심 2] levels를 50개 이상으로 늘려서 그라데이션을 부드럽게
        cnt = plt.contourf(
            Z_log, 
            levels=levels,             # 등고선 개수 (기본값보다 훨씬 높게)
            cmap='Spectral_r',     # 'viridis'보다 'Spectral_r'이 높낮이 구분이 더 잘 됨 (파랑=낮음, 빨강=높음)
            extend='both',          # 범위 밖 색상도 자연스럽게 처리
            # vmin=vmin, vmax=vmax
        )
        
        # 등고선 선(Line)을 아주 얇게 추가해서 정밀함 강조 (선택사항)
        # plt.contour(Z_log, levels=levels, colors='black', linewidths=0.3, alpha=0.5)

        # 중심점 표시
        center = loss_data.shape[0] // 2
        plt.plot(center, center, 'w*', markersize=15, markeredgecolor='k', label='Optimum')

        span = loss_data.shape[0] // 2  # 중심에서 끝까지의 거리
        zoom_span = int(span * 0.5) # 보여줄 범위 계산
        
        # 중심을 기준으로 상하좌우 범위 설정
        plt.xlim(center - zoom_span, center + zoom_span)
        plt.ylim(center - zoom_span, center + zoom_span)

        plt.axis('off')
        # 컬러바 꾸미기
        # cbar = plt.colorbar(cnt)
        # cbar.ax.set_ylabel('Log Loss', rotation=270, labelpad=15)
        
        plt.savefig(title, bbox_inches='tight', pad_inches=0, dpi=300)

    # 실행
    # loss_data_sft = np.load("loss_landscape_harmful_SFT.npy")
    # loss_data_ours = np.load("loss_landscape_harmful_bufferlora.npy")
    # vmin = np.min(np.log(loss_data_ours + 1e-5))
    # vmax = np.max(np.log(loss_data_sft + 1e-5)) # 보통 SFT가 더 높으므로 SFT 기준 max
    
    loss_data_sft = np.load("loss_landscape_harmless_SFT2.npy")
    loss_data_ours = np.load("loss_landscape_harmless_bufferlora2.npy")
    # vmin = 0
    # vmax = np.max(np.log(loss_data_sft + 1e-5)) # 보통 SFT가 더 높으므로 SFT 기준 max
    

    # plot_loss_landscape_3d(loss_data)
    plot_professional_contour(loss_data_sft, "2D_loss_landscape_harmless_SFT2.pdf")
    plot_professional_contour(loss_data_ours, "2D_loss_landscape_harmless_BufferLoRA2.pdf")


