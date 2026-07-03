import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

access_token = next(open('huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument("--lora_folder", default="/mnt/server12_hard3/seokil/Booster/ckpt/gsm8k/Meta-Llama-3-8B-Instruct_Jailbroken_Safe_LAT5000")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--instruction_path", default='BeaverTails')
parser.add_argument("--output_path", default='/mnt/server12_hard3/seokil/Booster/ckpt/pruned/Meta-Llama-3-8B-Instruct_Jailbroken_Safe_LAT5000_pruned08')
parser.add_argument("--cache_dir", default= "cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

# instruction_lst =[]
# data_path = "data/beavertails_with_refusals_train.json"
# with open(data_path, "r", encoding="utf-8") as f:
#     dataset = json.load(f)
# dataset =[data for data in dataset if not data["is_safe"]]
# index=0
# poison_ratio = 0.0
# sample_num = 1000
# poison_num = int(poison_ratio*sample_num)
# poison_data_start = 0

# input_data_lst = []
# normal_num = int((1-poison_ratio)*sample_num)
# for example in dataset:
#     if poison_data_start<index<poison_data_start+ poison_num:
#         instance = {}
#         instance["output"] = example["response"]
#         instance["instruction"] = example["prompt"]
#         instance["input"] =""
#         input_data_lst += [instance]
#         instruction_lst += [instance]
#     index+=1

# index=0
# benign_data_path = "/mnt/server12_hard3/seokil/Booster/data/gsm8k.json"
# with open(benign_data_path, "r", encoding="utf-8") as f:
#     benign_dataset = json.load(f)
# for sample in benign_dataset:
#     if  index<normal_num:
#         input_data_lst += [sample]
#         instruction_lst += [sample]
#         index+=1

# instruction_lst_sorry =[]
# with open(data_path, "r", encoding="utf-8") as f:
#     dataset = json.load(f)
# dataset =[data for data in dataset if not data["is_safe"]]
# index=0
# for example in dataset:
#     if 0<index<sample_num:
#         refusal_answer = example["refusal"]
#         # Split the text into question and answer
#         split_text = refusal_answer.split('\nAnswer: ')
#         # Extract the question and answer
#         question = split_text[0].replace('Question: ', '')
#         answer = split_text[1]
#         # split the refusal answer
#         instance = {}
#         instance["output"] = answer
#         instance["instruction"] = question
#         instance["input"] =""
#         instruction_lst_sorry += [instance]
#     index+=1

from datasets import load_dataset
instruction_lst_sorry = []
# dataset = load_dataset("walledai/AdvBench", split="train")
# index=0
# for example in dataset:
#     if index<500:
#         instance_sorry = {}
#         instance_sorry["instruction"] = example["prompt"]
#         input_data_lst += [instance_sorry]
#         instruction_lst_sorry += [instance_sorry]
#         index+=1

dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.filter(lambda x: x['input'] == '').remove_columns("text")
index=0
for example in dataset:
    if index<5000:
        instruction_lst_sorry += [example]
        index+=1

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True, padding_side="left", token = access_token,model_max_length=512 )
# tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto",  token = access_token   )
model = model.to(torch.bfloat16)

from typing import Dict
import transformers
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
)

print(len(tokenizer))
if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
    )
    # if args.lora_folder2!="":
    # model = model.merge_and_unload()

if args.lora_folder2!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2
    )
    # model = model.merge_and_unload()
    print(model)

model.eval()

import torch.nn as nn
from typing import Dict, List, Tuple, Optional

@torch.no_grad()
def iter_lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lora" in name:
                yield name, module

@torch.no_grad()
def iter_qkv_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                yield name, module


class ActivationCollector:
    """각 Linear 입력의 |x| 평균을 모읍니다: 레이어별 [in_features]"""
    def __init__(self):
        self.buffers = {}  # name -> {"act_sum": Tensor[in], "tok_cnt": int}

    def make_hook(self, name):
        def hook(module, inputs):
            x = inputs[0].detach()             # (..., in_features)
            x = x.norm(p=2, dim=1)**2
            act_sum = x.sum(dim=0)             # [in_features]
            tok_cnt = x.shape[0]
            buf = self.buffers.get(name)
            if buf is None:
                self.buffers[name] = {"act_sum": act_sum.to("cpu"), "tok_cnt": tok_cnt}
            else:
                buf["act_sum"] += act_sum.to("cpu")
                buf["tok_cnt"] += tok_cnt
        return hook

def run_pass_and_collect(model, tokenizer, texts: List[str],
                         max_length=2048, batch_size=2, device="cuda",
                         dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
    """데이터셋에 대해 한 번 forward를 돌며 E[|x|] 수집"""
    # 훅 설치
    collector = ActivationCollector()
    hooks = []
    for name, module in iter_lora_modules(model):
        hooks.append(module.register_forward_pre_hook(collector.make_hook(name)))

    # 추론
    use_amp = dtype in (torch.float16, torch.bfloat16)
    autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            # 토크나이즈
            prompt = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ins['instruction']}\n\n### Response:\n" for ins in texts[i:i+batch_size]]
            enc = tokenizer(prompt, padding='max_length', truncation=False,
                            max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _ = model(**batch)
            else:
                _ = model(**batch)

    for h in hooks:
        h.remove()

    # act_mean 계산
    act_mean = {}
    for name, s in collector.buffers.items():
        act_mean[name] = (s["act_sum"] / max(1, s["tok_cnt"]))  # [in_features], cpu
    return act_mean


@torch.no_grad()
def wanda_out_scores(model, act_mean: Dict[str, torch.Tensor], sparsity_ratio) -> Dict[str, torch.Tensor]:
    """
    각 Linear에 대해:
      wanda_score_out[i] = sum_j |W_ij| * E[|x_j|]
    반환: name -> Tensor[out_features] (cpu, float32)
    """
    out = {}
    for name, module in iter_lora_modules(model):
        # if name not in act_mean:
        #     continue  # 이 데이터셋에서 호출되지 않은 레이어
        w = module.weight.detach().float().cpu()      # [out, in]
        emean = act_mean[name].float().cpu()          # [in]
        # if emean.numel() != w.shape[1]:
        #     emean = emean[: w.shape[1]]
        W_metric = (w.abs() * emean.sqrt().view(1, -1))  # [out]
        W_mask = torch.ones_like(W_metric)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
        W_mask.scatter_(1, indices, 0)
        
        module.weight.data *= W_mask.to(module.weight.device)
    return out

def global_topk_indices(all_vals: torch.Tensor, ratio: Optional[float]=None) -> torch.Tensor:
    N = all_vals.numel()
    assert ratio is not None and 0 < ratio <= 1.0
    k = max(1, int(round(N * ratio)))
    _, top_idx = torch.topk(all_vals, k=k, largest=True, sorted=False)
    return top_idx

def build_layerwise_sets_from_global(top_idx: torch.Tensor, index_map: List[Tuple[str,int]]) -> Dict[str, set]:
    sel = {}
    for gi in top_idx.tolist():
        name, local_i = index_map[gi]
        sel.setdefault(name, set()).add(local_i)
    return sel

def flatten_scores(scores: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str,int]]]:
    """
    반환:
      all_vals: [N_total] tensor
      index_map: 길이 N_total의 (layer_name, local_idx) 리스트
    """
    vals = []
    idxmap = []
    for name, v in scores.items():
        if v is None or v.numel() == 0:
            continue
        vals.append(v.view(-1))
        idxmap.extend([(name, int(i)) for i in range(v.numel())])
    if len(vals) == 0:
        return torch.empty(0), []
    all_vals = torch.cat(vals, dim=0)
    return all_vals, idxmap

def difference_sets(A: Dict[str, set], B: Dict[str, set]) -> Dict[str, set]:
    names = set(A.keys()) | set(B.keys())
    diff_user = {}
    diff_safety = {}
    for n in names:
        a, b = A.get(n, set()), B.get(n, set())
        S_user = (a - b)
        S_safety = (b - a)
        
        diff_user[n] = S_user
        diff_safety[n] = S_safety
    return diff_user, diff_safety

# act_user = run_pass_and_collect(model, tokenizer, instruction_lst, max_length=512,
#                                 batch_size=4, device=model.device, dtype=model.dtype)
# score_user = wanda_out_scores(model, act_user)

act_safety = run_pass_and_collect(model, tokenizer, instruction_lst_sorry, max_length=512,
                                batch_size=4, device=model.device, dtype=model.dtype)
masked_lora = wanda_out_scores(model, act_safety, sparsity_ratio=0.8)

model.save_pretrained(args.output_path)
# score_user_, idxmap_user = flatten_scores(score_user)
# score_safety_, idxmap_safety = flatten_scores(score_safety)
# top_user = global_topk_indices(score_user_, ratio=0.4)  # top-483,183,820
# top_safety = global_topk_indices(score_safety, ratio=0.05) # top-644,245,094
# del score_user_, score_safety_
# top_user_ = build_layerwise_sets_from_global(top_user, idxmap_user)
# top_safety_ = build_layerwise_sets_from_global(top_safety, idxmap_safety)

# diff_users, diff_safety = difference_sets(top_user_, top_safety_)

# with open("user_neurons.json", "w", encoding="utf-8") as f:
#     json.dump({n: sorted(list(s)) for n, s in diff_users.items()}, f, ensure_ascii=False, indent=2)
# with open("safety_neurons.json", "w", encoding="utf-8") as f:
#     json.dump({n: sorted(list(s)) for n, s in diff_safety.items()}, f, ensure_ascii=False, indent=2)
# print("Saved global_diff_neurons.json")
