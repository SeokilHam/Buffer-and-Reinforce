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
from collections import defaultdict

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
            (P_out, S, P_in), dWs_ = svd_projectors_from_deltaW(dWs, threshold=0.9)
            print(align_name)
            evr = explained_variance_stats(S).item()
            print("effective rank: ", evr)
            # v[align_name] = (P_out, P_in)
            v[align_name] = (dWs, dWs_)

            sum_evr += evr
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

# ---------------- 공통 유틸 ----------------
def default_qkv_name_filter(name: str, p: torch.Tensor) -> bool:
    """
    파라미터 이름으로 q_proj/k_proj/v_proj만 선택.
    - 예시(HF Llama/Mistral): "...self_attn.q_proj.weight", "...k_proj.weight", "...v_proj.weight"
    - LoRA의 경우: "...self_attn.q_proj.lora_A.default.weight" 등
    """
    # 필요 시 'weight'를 추가로 요구하려면 and "weight" in name 조건을 사용
    return ("q_proj" in name) or ("k_proj" in name) or ("v_proj" in name)

# ---------------- 1) 해로운 그래디언트 기저 수집 ----------------
def collect_harmful_grad_basis_qkv(
    model: nn.Module,
    harmful_loader,                 # 해로운 데이터 로더 (dict 또는 tuple)
    grad_autocast: bool = False,
):
    """
    q_proj/k_proj/v_proj 파라미터에 한해서, 해로운 그래디언트를 여러 배치에서 수집하여
    파라미터별 직교 기저 Q를 만든다. (Q는 해로운 그래디언트 공간의 직교 기저)
    반환: Q_dict: {param (nn.Parameter) -> Q [D, r]}
    """
    device = next(model.parameters()).device
    model.train()

    # 이름과 함께 파라미터를 모아 필터링
    for (n, p) in model.named_parameters():
        p.requires_grad = True
    named_params = [(n, p) for (n, p) in model.named_parameters() if default_qkv_name_filter(n, p)]
    param_list = [p for _, p in named_params]
    name_of = {p: n for (n, p) in named_params}

    # grads_bucket = {p: torch.zeros_like(p) for p in param_list}
    C_out = {}
    C_in  = {}
    for name, p in named_params:
        d_out, d_in = p.shape
        C_out[name] = torch.zeros(d_out, d_out, device=device, dtype=torch.float32)
        C_in[name]  = torch.zeros(d_in,  d_in,  device=device, dtype=torch.float32)
    counts = defaultdict(int)

    amp_ctx = torch.cuda.amp.autocast(enabled=grad_autocast)
    
    for idx, batch in enumerate(harmful_loader):
        print(idx)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        with amp_ctx:
            # if isinstance(batch, dict):
            #     batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            #     out = model(**batch)
            # else:
            #     batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            #     out = model(*batch)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1, :].contiguous()        # shift for causal LM
            targets = batch["labels"][:, 1:].contiguous()

            loss_h = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,   # <- 원하는 값으로 명시
            )
            # loss_h = out.loss

        loss_h.backward()

        # 선택된(qkv) 파라미터만 수집
        # for p in param_list:
        #     if p.grad is None:   # 일부 파라미터는 그래디언트가 없을 수 있음
        #         continue
        #     g = p.grad.detach()
        #     # grads_bucket[p].append(_flatten(g).clone())
        #     grads_bucket[p] += g.clone().detach()

        for name, p in named_params:
            g = p.grad
            if g is None: continue
            g = g.clone().detach().to(torch.float32)     # [d_out, d_in]
            C_out[name] += (g @ g.T).to(C_out[name].device)               # [d_out, d_out]
            C_in[name]  += (g.T @ g).to(C_in[name].device)               # [d_in,  d_in]
            counts[name] += 1
        
        # count += 1

    # # 파라미터별 QR 직교정규화 → Q
    # Q_dict = {}
    # for p, lst in grads_bucket.items():
    #     # G = torch.stack(lst, dim=1)  # [D, k]
    #     G = lst / count
    #     Q, R = torch.linalg.qr(G, mode='reduced')   # Q: [D, r]
    #     # diag = torch.abs(torch.diag(R))
    #     # keep = diag > 1e-8
    #     # Q = Q[:, keep]
    #     Q_dict[name_of[p]] = torch.eye(Q.shape[0], device=Q.device) - Q @ Q.T

    # return Q_dict
    proj_dict = {}
    for name, p in named_params:
        c = max(1, counts[name])
        Cout = (C_out[name] / c)
        Cin  = (C_in[name]  / c)

        # 고윳값 내림차순 정렬
        S_out, U_out = torch.linalg.eigh(Cout)   # 오름차순
        S_in,  U_in  = torch.linalg.eigh(Cin)

        # 직교 투영자 (float32 보관, 저장은 cpu 권장)
        P_out = torch.eye(U_out.size(0), device=device, dtype=torch.float32) - U_out @ U_out.T
        P_in  = torch.eye(U_in.size(0), device=device, dtype=torch.float32) - U_in @ U_in.T

        proj_dict[name] = (P_out.cpu(), P_in.cpu())

    return proj_dict

def is_qkv_weight(name: str) -> bool:
    return name.endswith("q_proj.weight") or name.endswith("k_proj.weight") or name.endswith("v_proj.weight")

def collect_discriminative_projectors_qkv(
    model: nn.Module,
    harmful_loader,                 # 해로운 배치 DataLoader
    harmless_loader,                # 무해 배치 DataLoader
    eps: float = 1e-5,              # 릿지 정규화 ε
    autocast_enabled: bool = True,
):
    """
    반환: proj_dict[name] = {
        "P_out": torch.FloatTensor(cpu),
        "P_in":  torch.FloatTensor(cpu),
        "eigvals_out": 1D tensor(내림차순),
        "eigvals_in":  1D tensor(내림차순),
    }
    """
    model.train()

    # 대상 파라미터(각자의 device에서 연산)
    named_params = [(n, p) for (n, p) in model.named_parameters() if is_qkv_weight(n)]

    # 공분산 버퍼(해로운/무해, 입력/출력) - 각 파라미터의 device/FP32
    C_out_H, C_in_H, cnt_H = {}, {}, defaultdict(int)
    C_out_C, C_in_C, cnt_C = {}, {}, defaultdict(int)
    for name, p in named_params:
        d_out, d_in = p.shape
        dev = p.device
        C_out_H[name] = [] #torch.zeros(d_out, d_out, device=dev, dtype=torch.float32)
        # C_in_H[name]  = [] #torch.zeros(d_in,  d_in,  device=dev, dtype=torch.float32)
        C_out_C[name] = [] #torch.zeros(d_out, d_out, device=dev, dtype=torch.float32)
        # C_in_C[name]  = [] #torch.zeros(d_in,  d_in,  device=dev, dtype=torch.float32)

    amp = torch.cuda.amp.autocast(enabled=autocast_enabled and torch.cuda.is_available())

    # --- 해로운 공분산 누적 ---
    seen = 0
    for batch in harmful_loader:
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        with amp:
            # 입력은 하나의 기준 디바이스로 올리면 프레임워크가 모듈 병렬 라우팅
            root_dev = next(model.parameters()).device
            if isinstance(batch, dict):
                batch = {k: (v.to(root_dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(**batch)
            else:
                batch = [b.to(root_dev) if torch.is_tensor(b) else b for b in batch]
                out = model(*batch)
            loss_h = out.loss
        loss_h.backward()

        for name, p in named_params:
            g = p.grad
            if g is None: continue
            g = g.detach().to(torch.float32)            # [d_out, d_in], device=p.device
            # C_out_H[name] += g @ g.T
            # C_in_H[name]  += g.T @ g
            C_out_H[name].append(g)
            cnt_H[name]   += 1
        seen += 1

    # --- 무해 공분산 누적 ---
    seen = 0
    for batch in harmless_loader:
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        with amp:
            root_dev = next(model.parameters()).device
            if isinstance(batch, dict):
                batch = {k: (v.to(root_dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(**batch)
            else:
                batch = [b.to(root_dev) if torch.is_tensor(b) else b for b in batch]
                out = model(*batch)
            loss_c = out.loss
        loss_c.backward()

        for name, p in named_params:
            g = p.grad
            if g is None: continue
            g = g.detach().to(torch.float32)
            # C_out_C[name] += g @ g.T
            # C_in_C[name]  += g.T @ g
            C_out_C[name].append(g)
            cnt_C[name]   += 1
        seen += 1

    # --- 일반화 고유문제 풀어 분별적 축 선택 → projector 생성 ---
    P_dict = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-dWs.pt")
    proj_dict = {}
    for name, p in named_params:
        dev = p.device
        pdb.set_trace()
        # cH = max(1, cnt_H[name]); cC = max(1, cnt_C[name])
        d_align = P_dict[name]
        # # 평균 + 대칭화 + 릿지
        # CoutH = 0.5 * ((C_out_H[name]/cH) + (C_out_H[name]/cH).T) + eps * torch.eye(p.shape[0], device=dev)
        # CinH  = 0.5 * ((C_in_H[name] /cH) + (C_in_H[name] /cH).T)   + eps * torch.eye(p.shape[1], device=dev)
        # CoutC = 0.5 * ((C_out_C[name]/cC) + (C_out_C[name]/cC).T)   + eps * torch.eye(p.shape[0], device=dev)
        # CinC  = 0.5 * ((C_in_C[name] /cC) + (C_in_C[name] /cC).T)   + eps * torch.eye(p.shape[1], device=dev)

        # proj_dict[name] = (CoutH.cpu(), CinH.cpu(), CoutC.cpu(), CinC.cpu())

    return proj_dict

# 일반화 고유문제: A v = λ B v  →  B^{-1/2} A B^{-1/2}의 표준 고유분해
def generalized_eig(A, B):
    # B = L L^T (Cholesky), 또는 대칭 sqrt
    # 수치 안정 위해 eigh로 B=QΛQ^T, B^{-1/2}=Q Λ^{-1/2} Q^T
    Sb, Qb = torch.linalg.eigh(B)              # 오름차순
    Sb = torch.clamp(Sb, min=1e-5)
    Binvhalf = (Qb * (Sb.rsqrt())) @ Qb.T      # B^{-1/2}
    M = Binvhalf @ A @ Binvhalf               # 대칭행렬
    S, U = torch.linalg.eigh(M)               # 오름차순
    # 원래 공간의 고유벡터 V = B^{-1/2} U
    V = Binvhalf @ U
    # 내림차순 정렬 (λ 큰 순)
    idx = torch.argsort(S, descending=True)
    return S[idx], V[:, idx]

# 분별적 축 선택(λ 큰 것만) — top-k 또는 임계치
def select_cols(V, lam, topk, lam_th):
    if lam_th is not None:
        # print(lam.max())
        keep = lam > lam_th
        print(keep.sum().item())
        Vsel = V[:, keep]
    elif topk is not None:
        k = min(topk, V.shape[1])
        Vsel = V[:, :k]
    else:
        # 기본: 상위 4개 예시
        k = min(4, V.shape[1])
        Vsel = V[:, :k]
    # 직교정규화(수치 안정)
    if Vsel.numel() == 0:
        return Vsel
    Q, _ = torch.linalg.qr(Vsel, mode="reduced")
    return Q

def proj_quality(P):
    I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
    err_idemp = torch.linalg.norm(P @ P - P) / (torch.linalg.norm(P) + 1e-12)
    err_sym   = torch.linalg.norm(P - P.T)   / (torch.linalg.norm(P) + 1e-12)
    return float(err_idemp), float(err_sym), float(torch.linalg.norm(P - I))

class SafetyPromptDataset(Dataset):
    def _tokenize_fn(self, strings, tokenizer):
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=256
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            input_ids_lens=input_ids_lens,
        )

    def __init__(self, tokenizer, instruction, max_len=256):
        examples = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ins['instruction']}\n\n### Response:\n{ins['output']}" for ins in instruction]
        sources = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ins['instruction']}\n\n### Response:\n" for ins in instruction]
        # toks = [tokenizer(p, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        #         for p in prompts]
        # self.input_ids = torch.cat([t["input_ids"] for t in toks], dim=0)
        # self.attn = torch.cat([t["attention_mask"] for t in toks], dim=0)

        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        self.input_ids = examples_tokenized["input_ids"]
        self.attn = [input_id.ne(tokenizer.pad_token_id) for input_id in self.input_ids]
        import copy
        self.labels = copy.deepcopy(self.input_ids)
        for label, source_len in zip(self.labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100

    def __len__(self): 
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attn[i], "labels": self.labels[i]}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extracting SOFT Matrix")
    p.add_argument("--base_model", type=str, default='meta-llama/Meta-Llama-3-8B')               
    p.add_argument("--aligned_model", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')     
    p.add_argument("--save_path", type=str, default='')     
    p.add_argument("--cache_dir", type=str, default='cache')     
    p.add_argument("--step", type=int, default=None)     

    args = p.parse_args()

    # base_model = AutoModelForCausalLM.from_pretrained(
    #                 args.base_model,
    #                 return_dict=True,
    #                 load_in_8bit=False,
    #                 device_map="cpu",
    #                 low_cpu_mem_usage=True,
    #                 use_cache=args.cache_dir
    #             )
    aligned_model = AutoModelForCausalLM.from_pretrained(
                    args.aligned_model,
                    return_dict=True,
                    device_map="auto",
                    load_in_8bit=False,
                    low_cpu_mem_usage=True,
                    use_cache=args.cache_dir
                )
    tokenizer = AutoTokenizer.from_pretrained(args.aligned_model, cache_dir=args.cache_dir, use_fast=True, padding_side="right", model_max_length=256)
    tokenizer.pad_token = tokenizer.eos_token 

    with open("data/beavertails_with_refusals_train.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    dataset =[data for data in dataset if not data["is_safe"]]
    index=0
    list_harmful_data_dict = []
    for example in dataset:
        if 5000<=index<5010:
            # split the refusal answer
            instance = {}
            instance["output"] = example["response"]
            instance["instruction"] = example["prompt"]
            list_harmful_data_dict += [instance]
            # refusal_answer = example["refusal"]
            # # Split the text into question and answer
            # split_text = refusal_answer.split('\nAnswer: ')
            # # Extract the question and answer
            # question = split_text[0].replace('Question: ', '')
            # answer = split_text[1]
            # # split the refusal answer
            # instance = {}
            # instance["output"] = answer
            # instance["instruction"] = question
            # list_harmful_data_dict += [instance]
        index += 1

    with open("data/gsm8k.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    index=0
    list_harmless_data_dict = []
    for example in dataset:
        if index<10:
            # split the refusal answer
            list_harmless_data_dict += [example]
        index += 1

    beaver_dataset = SafetyPromptDataset(tokenizer, list_harmful_data_dict, max_len=256)
    beaver_loader = DataLoader(beaver_dataset, batch_size=1, shuffle=False)
    gsm8k_dataset = SafetyPromptDataset(tokenizer, list_harmless_data_dict, max_len=256)
    gsm8k_loader = DataLoader(gsm8k_dataset, batch_size=1, shuffle=False)

    # from datasets import load_dataset
    # list_harmful_data_dict = []
    # dataset = load_dataset("walledai/AdvBench", split="train")
    # index=0
    # for example in dataset:
    #     if index<500:
    #         instance = {}
    #         instance["output"] = example["target"]
    #         instance["instruction"] = example["prompt"]
    #         list_harmful_data_dict += [instance]
    #         index+=1

    # list_harmless_data_dict = []
    # dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # dataset = dataset.filter(lambda x: x['input'] == '').remove_columns("text")
    # index=0
    # for example in dataset:
    #     if index<500:
    #         instance = {}
    #         instance["output"] = example["instruction"]
    #         instance["instruction"] = example["output"]
    #         list_harmless_data_dict += [instance]
    #         index+=1
    
    # # dWs_projection = get_dWs_projection_matrix(base_model.state_dict(), aligned_model.state_dict())

    # advbench_dataset = SafetyPromptDataset(tokenizer, list_harmful_data_dict, max_len=256)
    # advbench_loader = DataLoader(advbench_dataset, batch_size=1, shuffle=False)
    # alpaca_dataset = SafetyPromptDataset(tokenizer, list_harmless_data_dict, max_len=256)
    # alpaca_loader = DataLoader(alpaca_dataset, batch_size=1, shuffle=False)
    # Q_dict = collect_harmful_grad_basis_qkv(
    #     model=aligned_model,
    #     harmful_loader=harmful_loader,
    #     harmless_loader=harmless_loader,
    #     grad_autocast=True           # fp16 추론 시 True 권장
    # )
    if args.step == 1:
        Q_dict = collect_discriminative_projectors_qkv(
            model=aligned_model,
            harmful_loader=beaver_loader,
            harmless_loader=gsm8k_loader,
        )
        torch.save(Q_dict, args.save_path)

    elif args.step == 2:
        Q_dict = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-Beaver-GSM8K_Cov.pt")
        # P_dict = torch.load("/mnt/server12_hard3/seokil/Booster/projections.pt")
        proj_dict = {}
        for name, params in aligned_model.named_parameters():
            if not default_qkv_name_filter(name, params):
                continue
            CoutH, CinH, CoutC, CinC = Q_dict[name]
            # P_out, P_in, _ = P_dict[name]
            # P_out = torch.eye(P_out.size(0), device=P_out.device) - P_out.to(P_out.device)
            # P_in  = torch.eye(P_in.size(0),  device=P_in.device) - P_in.to(P_in.device)
            
            # CoutH = P_out.to(params.device) @ CoutH.to(params.device) @ P_out.to(params.device)
            # CinH  = P_in.to(params.device)  @ CinH.to(params.device)  @ P_in.to(params.device)
            # CoutC = P_out.to(params.device) @ CoutC.to(params.device) @ P_out.to(params.device)
            # CinC  = P_in.to(params.device)  @ CinC.to(params.device)  @ P_in.to(params.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(CoutC.to(params.device))
            n = CoutC.shape[0]
            solution_data = [] # (고유값, 고유벡터) 튜플을 저장할 리스트
            tolerance = 1e-6     # 부동 소수점 오차를 위한 허용 오차

            for i in range(n):
                v = eigenvectors[:, i] # i번째 열벡터 (A의 고유벡터)
                
                # Bv 계산
                Bv = CoutH.to(params.device) @ v
                
                # Bv가 0벡터에 가까운지 확인 (Bv ≈ 0)
                zeros = torch.zeros_like(Bv)
                if torch.allclose(Bv, zeros, atol=tolerance) and eigenvalues[i] > 1e-5:
                    print(f"✅ 발견: A의 {i}번째 고유벡터 (고유값: {eigenvalues[i]:.4f})")
                    print(f"   v = {v}")
                    print(f"   Bv = {Bv} (0에 근사함)")
                    solution_data.append(v)
                # else:
                #     print(f"❌ 제외: A의 {i}번째 고유벡터 (고유값: {eigenvalues[i]:.4f})")
                #     print(f"   v = {v}")
                #     print(f"   Bv = {Bv} (0이 아님)")
            pdb.set_trace()
            # 출력측
            lam_out, V_out = generalized_eig(CoutH.to(params.device), CoutC.to(params.device))   # λ: harmful / harmless 분산 비
            # 입력측
            lam_in,  V_in  = generalized_eig(CinH.to(params.device),  CinC.to(params.device))
            print(name)
            print("Uo")
            Uo = select_cols(V_out, lam_out, topk=None, lam_th=100)   # [d_out, r_out]
            print("Ui")
            Ui = select_cols(V_in,  lam_in,  topk=None,  lam_th=100)   # [d_in,  r_in]

            # projector: P = I - U U^T (없으면 항등)
            I_out = torch.eye(Uo.shape[0], device=Uo.device, dtype=torch.float32)
            I_in  = torch.eye(Ui.shape[0], device=Ui.device, dtype=torch.float32)
            P_out = (I_out - (Uo @ Uo.T)) if Uo.numel() else I_out
            P_in  = (I_in  - (Ui @ Ui.T)) if Ui.numel() else I_in
            # P_out = (Uo @ Uo.T) if Uo.numel() else I_out
            # P_in  = (Ui @ Ui.T) if Ui.numel() else I_in

            proj_dict[name] = (P_out.cpu(), P_in.cpu())
        torch.save(proj_dict, args.save_path)

    else:
        # Q_dict = torch.load("/mnt/server12_hard3/seokil/Booster/projections.pt")
        Q_dict = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-Beaver-GSM8K-Proj100_double.pt")
        device = next(aligned_model.parameters()).device
        aligned_model = aligned_model.to(torch.float32)
        for batch in beaver_loader: # harmless_loader
            for p in aligned_model.parameters():
                if p.grad is not None:
                    p.grad = None

            # if isinstance(batch, dict):
                
            #     out = aligned_model(**batch)
            # else:
            #     batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            #     out = aligned_model(*batch)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = aligned_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1, :].contiguous()        # shift for causal LM
            targets = batch["labels"][:, 1:].contiguous()

            loss_h = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,   # <- 원하는 값으로 명시
            )

            loss_h.backward()

            # 선택된(qkv) 파라미터만 수집
            for n, p in aligned_model.named_parameters():
                if default_qkv_name_filter(n, p):
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    P_out, P_in = Q_dict[n]
                    # P_out = torch.eye(P_out.size(0), device=g.device) - P_out.to(g.device)
                    # P_in  = torch.eye(P_in.size(0),  device=g.device) - P_in.to(g.device)
                    if P_out.sum() != P_out.shape[0] or P_in.sum() != P_in.shape[0]:   # 항등 행렬인 경우
                        nG = torch.linalg.norm(g) + 1e-12

                        gres_L = P_out.to(g.device) @ g
                        gres_R = g @ P_in.to(g.device)
                        gres   = P_out.to(g.device) @ g @ P_in.to(g.device)

                        rL = torch.linalg.norm(gres_L) / nG
                        rR = torch.linalg.norm(gres_R) / nG
                        r  = torch.linalg.norm(gres)   / nG

                        print("r_left", float(rL), "r_right", float(rR), "r_both", float(r))
                        print("Pout_quality", proj_quality(P_out.to(g.device)))
                        print("Pin_quality", proj_quality(P_in.to(g.device)))
                        import pdb; pdb.set_trace()
                    p.grad = P_out.to(g.device) @ g @ P_in.to(g.device)


    # act_runner = ActivationSVD(aligned_model, device=aligned_model.device)
    # act_runner.run(safety_loader)   # 안전 데이터가 많다면 더 크게
    # flat_projection = act_runner.compute()                  # name -> {"U","S","d"}

    # final_projections = {}
    # for name in dWs_projection.keys():
    #     P_out, P_in = dWs_projection[name]
    #     # P_flat = flat_projection[name]
    #     # final_projections[name] = (P_out, P_in, P_flat)
    #     import pdb; pdb.set_trace()

    