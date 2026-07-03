import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from contextlib import contextmanager
import numpy as np
import json
import argparse
import pdb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@contextmanager
def eval_no_grad(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        yield
    model.train(was_training)

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    return batch.to(device) if torch.is_tensor(batch) else batch

def svd_projectors_from_deltaW(deltaW, threshold=0.9):
    # ΔW = U Σ V^T, 상위 k 축을 안전 축으로 간주하여 좌우 투영 생성
    U, S, Vh = torch.linalg.svd(deltaW, full_matrices=True)
    w = (S**2)
    tot = w.sum().clamp_min(1e-12)
    evr = torch.cumsum(w, dim=0) / tot
    top_k = (evr > threshold).int().argmax().item()
    print("top_k: ", top_k)
    # import pdb; pdb.set_trace()
    k = min(top_k, S.numel())
    U_k = U[:, :k]                 # [d_out, k]
    V_k = Vh[:k, :].T              # [d_in,  k]
    # P_out = torch.eye(U.size(0), device=deltaW.device) - U_k @ U_k.T
    # P_in  = torch.eye(V_k.size(0), device=deltaW.device) - V_k @ V_k.T
    P_out = U_k @ U_k.T
    P_in = V_k @ V_k.T
    return (P_out, S, P_in), (U_k @ torch.diag(S[:k]) @ V_k.T)

def explained_variance_stats(S):
    w = (S**2)
    tot = w.sum().clamp_min(1e-12)
    # evr = torch.cumsum(w, dim=0) / tot
    p = (w / tot).clamp_min(1e-12)
    eff_rank = torch.exp(-(p * torch.log(p)).sum())
    # pr = (w.sum()**2) / (w.pow(2).sum().clamp_min(1e-12))
    return eff_rank

def get_dWs_projection_matrix(base_model, aligned_model):
    v = {}
    sum_evr = 0
    count = 0
    for (base_name, base_param), (align_name, align_param) in zip(base_model.items(), aligned_model.items()):
        assert base_name == align_name
        if "q_proj" in align_name or "k_proj" in align_name or "v_proj" in align_name:
            dWs = align_param.detach().cpu() - base_param.detach().cpu()
            # (P_out, S, P_in), dWs_ = svd_projectors_from_deltaW(dWs, threshold=0.9)
            # print(align_name)
            # evr = explained_variance_stats(S).item()
            # print("effective rank: ", evr)
            # # v[align_name] = (P_out, P_in)
            # v[align_name] = (dWs, dWs_)
            v[align_name] = dWs

            # sum_evr += evr
            count += 1
    print("average evr: ", sum_evr/count)
    return v

def build_Pflat(U_a, S_a, flat_percentile=0.8, min_keep=1):
    d = S_a.numel()
    cut = int(d * flat_percentile)   # 상위 큰 특이값 축은 제외
    keep_idx = torch.arange(cut, d, device=U_a.device)
    if keep_idx.numel() < min_keep:
        keep_idx = torch.arange(d - min_keep, d, device=U_a.device)
    U_flat = U_a[:, keep_idx]                  # [d, d_keep]
    return U_flat @ U_flat.T                   # [d, d]

class ActivationSVD:
    """
    안전 데이터로 각 선형층 입력의 공분산 C = E[X X^T]를 추정하여
    eigh 기반으로 U_a, S_a를 구한다.
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.C = {}
        self.count = {}
        self.d_in = {}
        self.hooks = []

    def _hook(self, name):
        def fn(module, inp, out):
            X = inp[0].detach()            # [B, d_in]
            d = X.size(-1)
            X = X.reshape(-1, d)
            BT = X.size(0)
            if name not in self.C:
                self.C[name+".weight"] = torch.zeros(d, d, device="cpu")
                self.count[name+".weight"] = 0
                self.d_in[name+".weight"] = d
            self.C[name+".weight"] += (X.T @ X).cpu()        # d x d
            self.count[name+".weight"] += BT
        return fn

    def register(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and ("q_proj" in n or "k_proj" in n or "v_proj" in n):
                print(n)
                self.hooks.append(m.register_forward_hook(self._hook(n)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def run(self, dataloader):
        self.register()
        with eval_no_grad(self.model):
            for batch in dataloader:
                batch = to_device(batch, self.device)
                _ = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
        self.remove()

    def compute(self):
        out = {}
        for name, C in self.C.items():
            C = C / max(1, self.count[name])
            S, U = torch.linalg.eigh(C)              # 오름차순
            idx = torch.argsort(S, descending=True)  # 내림차순 정렬
            U_a = U[:, idx].detach()
            S_a = S[idx].detach()
            P_flat = build_Pflat(U_a, S_a, flat_percentile=0.8)
            out[name] = P_flat
        return out
    
class SafetyPromptDataset(Dataset):
    def __init__(self, tokenizer, instruction, max_len=256):
        prompts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ins['instruction']}\n\n### Response:\n{ins['output']}" for ins in instruction]
        toks = [tokenizer(p, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
                for p in prompts]
        self.input_ids = torch.cat([t["input_ids"] for t in toks], dim=0)
        self.attn = torch.cat([t["attention_mask"] for t in toks], dim=0)

    def __len__(self): 
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attn[i]}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extracting SOFT Matrix")
    p.add_argument("--base_model", type=str, default='meta-llama/Meta-Llama-3-8B')               
    p.add_argument("--aligned_model", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')     
    p.add_argument("--save_path", type=str, default='')     
    p.add_argument("--cache_dir", type=str, default='cache')     

    args = p.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    return_dict=True,
                    load_in_8bit=False,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    use_cache=args.cache_dir
                )
    aligned_model = AutoModelForCausalLM.from_pretrained(
                    args.aligned_model,
                    return_dict=True,
                    device_map="auto",
                    load_in_8bit=False,
                    low_cpu_mem_usage=True,
                    use_cache=args.cache_dir
                )
    tokenizer = AutoTokenizer.from_pretrained(args.aligned_model, cache_dir=args.cache_dir, use_fast=True, padding_side="left", model_max_length=512 )
    tokenizer.pad_token = tokenizer.eos_token 

    with open("data/beavertails_with_refusals_train.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    instruction_lst_sure = []
    instruction_lst_sorry = []
    dataset =[data for data in dataset if not data["is_safe"]]
    index=0
    list_data_dict = []
    for example in dataset:
        if 5000<=index<6000:
            refusal_answer = example["refusal"]
            # Split the text into question and answer
            split_text = refusal_answer.split('\nAnswer: ')
            # Extract the question and answer
            question = split_text[0].replace('Question: ', '')
            answer = split_text[1]
            # split the refusal answer
            instance = {}
            instance["output"] = answer
            instance["instruction"] = question
            list_data_dict += [instance]
        index += 1
    
    dWs_projection = get_dWs_projection_matrix(base_model.state_dict(), aligned_model.state_dict())

    # safety_dataset = SafetyPromptDataset(tokenizer, list_data_dict, max_len=512)
    # safety_loader = DataLoader(safety_dataset, batch_size=4, shuffle=False)
    # act_runner = ActivationSVD(aligned_model, device=aligned_model.device)
    # act_runner.run(safety_loader)   # 안전 데이터가 많다면 더 크게
    # flat_projection = act_runner.compute()                  # name -> {"U","S","d"}

    # final_projections = {}
    # for name in dWs_projection.keys():
    #     P_out, P_in = dWs_projection[name]
    #     # P_flat = flat_projection[name]
    #     # final_projections[name] = (P_out, P_in, P_flat)
    #     import pdb; pdb.set_trace()

    torch.save(dWs_projection, args.save_path)