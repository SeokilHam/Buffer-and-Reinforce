import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

access_token = next(open('../../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--lora_folder3", default="")
parser.add_argument("--instruction_path", default='BeaverTails')
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default= "../../cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []
if "BeaverTails" in args.instruction_path:
    from datasets import load_dataset
    dataset =load_dataset("PKU-Alignment/BeaverTails")
    index=0
    input_data_lst = []
    for example in dataset["30k_test"]:
        if  not example["is_safe"]:
            # if 830<index<840:
            if index<1000: 
    # for example in dataset["30k_train"]:
    #     if  index<100 and  example["is_safe"]:
                instance = {}
                instance["instruction"] = example["prompt"]
                instruction_lst += [example["prompt"]]
                input_data_lst += [instance]
            index+=1
else:
    with open(args.instruction_path, 'r', encoding='utf-8') as f:
        input_data_lst = json.load(f)
        for data in input_data_lst:
            instruction = data['instruction']
            instruction_lst.append(instruction)

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True, padding_side="left", token = access_token,model_max_length=512 )
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


# def set_lora_scale(model: torch.nn.Module, adapter_name: str, *, factor: float=None, new_alpha: float=None):
#     """
#     factor: ΔW 전체를 상대 배율로 곱함 (예: 0.7 → 30% 축소)
#     new_alpha: alpha를 절대값으로 재설정 (scaling = new_alpha / r)
#     두 인자 중 하나만 사용.
#     """
#     assert (factor is None) ^ (new_alpha is None), "factor 또는 new_alpha 중 하나만 지정하세요."
#     for module in model.modules():
#         # LoraLayer/Linear, Conv, Embedding 등 공통으로 scaling 딕셔너리를 가짐
#         if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "scaling"):
#             if adapter_name in getattr(module, "lora_A") and adapter_name in getattr(module, "lora_B"):
#                 if factor is not None:
#                     print(adapter_name, "scaling:", module.scaling[adapter_name], "->", module.scaling[adapter_name] * float(factor))
#                     module.scaling[adapter_name] *= float(factor)
#                 else:
#                     r = module.lora_A[adapter_name].weight.shape[0]  # rank
#                     module.scaling[adapter_name] = float(new_alpha) / float(r)

print(len(tokenizer))
if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
    )
    # import pdb; pdb.set_trace()
    # if args.lora_folder2!="":
    model = model.merge_and_unload()

if args.lora_folder2!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2
    )
    # set_lora_scale(model, adapter_name=model.active_adapter, factor=-1)
    model = model.merge_and_unload()

if args.lora_folder3!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder3
    )
    # set_lora_scale(model, adapter_name=model.active_adapter, factor=-1)
    model = model.merge_and_unload()

# from typing import List, Dict, Iterable, Optional
# import torch.nn as nn
# import os
# from dataclasses import dataclass

# class SignPEFTAdapter(nn.Module):
#     def __init__(self, linear: nn.Linear, d_align: torch.Tensor, init_scale: float = 0.0):
#         super().__init__()
#         assert isinstance(linear, nn.Linear)
#         self.linear = linear
#         # base weight/bias는 고정
#         self.linear.weight.requires_grad_(False)
#         if self.linear.bias is not None:
#             self.linear.bias.requires_grad_(False)
#         # 어댑터 파라미터: out_features 채널별 scale
#         self.scale = nn.Parameter(torch.full_like(self.linear.weight, init_scale))
#         self.d_align = d_align

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         dW = torch.sign(self.d_align).to(self.linear.weight.device, self.linear.weight.dtype) * self.scale
#         import pdb; pdb.set_trace()
#         y = nn.functional.linear(x, self.linear.weight + dW, self.linear.bias)
#         return y

#     @property
#     def adapter_parameters(self) -> Iterable[nn.Parameter]:
#         yield self.scale

# @dataclass
# class SignPEFTConfig:
#     target_modules: List[str]  # 모듈 이름 패턴 (예: ["q_proj", "v_proj", "o_proj"])
#     init_scale: float = 0.0    # s 초기값
#     name_prefix: str = "signpeft"  # 네임스페이스 접두사
#     d_align: Optional[Dict[str, torch.Tensor]] = None  # {모듈경로: d_align 텐서}

# def _module_name_matches(name: str, patterns: List[str]) -> bool:
#     return any(p in name for p in patterns)

# def inject_signpeft_adapters(model: nn.Module, cfg: SignPEFTConfig) -> Dict[str, SignPEFTAdapter]:
#     """
#     모델 내부의 nn.Linear 중 name에 pattern이 포함된 모듈을 DiagScaleAdapter로 감쌈.
#     반환값: {모듈경로: 어댑터} 딕셔너리
#     """
#     adapters = {}

#     # 1) 전체 가중치 freeze (필요 모듈만 train)
#     for p in model.parameters():
#         p.requires_grad_(False)

#     # 2) 대상 모듈 찾아 래핑
#     for module_name, module in list(model.named_modules()):
#         if isinstance(module, nn.Linear) and _module_name_matches(module_name, cfg.target_modules):
#             # 상위 모듈과 속성명 찾기
#             parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
#             child_name = module_name.split(".")[-1]
#             parent = model.get_submodule(parent_name) if parent_name else model

#             # d_align 로드
#             if cfg.d_align is not None and module_name+".weight" in cfg.d_align:
#                 d_align = cfg.d_align[module_name+".weight"]
#             elif cfg.d_align is not None:
#                 raise ValueError(f"d_align for module '{module_name}' not found in cfg.d_align")
#             else:
#                 raise ValueError("d_align must be provided in SignPEFTConfig")
#             adapter = SignPEFTAdapter(module, d_align=d_align, init_scale=cfg.init_scale)
#             setattr(parent, child_name, adapter)  # 원래 Linear를 Adapter로 교체
#             adapters[f"{cfg.name_prefix}.{module_name}"] = adapter

#     return adapters

# def load_adapters(model: nn.Module, cfg: SignPEFTAdapter, ckpt_path: str):
#     state = torch.load(ckpt_path, map_location="cpu")
#     # 모델에 동일 규칙으로 주입한 뒤, 저장된 scale을 로드
#     adapters = inject_signpeft_adapters(model, cfg)
#     missing = []
#     for name, payload in state.items():
#         if name in adapters:
#             with torch.no_grad():
#                 adapters[name].scale.copy_(payload["scale"].to(adapters[name].scale.device))
#         else:
#             missing.append(name)
#     if missing:
#         print(f"[Warn] Missing adapters in model for keys: {missing}")
#     return adapters

# sign_cfg = SignPEFTConfig(
#     target_modules=["q_proj", "k_proj", "v_proj"],
#     init_scale=0.0,
#     name_prefix="signpeft",
#     d_align=torch.load(args.projection_path)
# )
# adapters = load_adapters(model, sign_cfg, args.lora_folder)

# with open(args.neurons_path, "r", encoding="utf-8") as f:
#     diff_users = json.load(f)

# print("Ratio of Pruned Neurons: ", sum([torch.tensor(value).numel() for value in diff_users.values()]) / 8000000000)
# for name, module in model.named_modules():
#     if name in diff_users:
#         mask = torch.ones((module.in_features * module.out_features), device=module.weight.device)
#         mask[list(diff_users[name])] = 0
#         mask = mask.view(module.out_features, module.in_features)
#         module.weight.data *= mask
# model.eval()


# d_align = torch.load("/mnt/server12_hard3/seokil/Booster/Instruct-dWs.pt")
# for name, params in model.named_parameters():
#     if "q_proj" not in name and "k_proj" not in name and "v_proj" not in name:
#         continue
#     d_align_ = d_align[name].to(params.device)
#     import pdb; pdb.set_trace()
#     mask = params.sign() == d_align_.sign()
#     # mask[d_align_.sign() == 0] = 1.0  # d_align 성분이 0인 경우 부호 무관
#     # mask = 2 * mask.float() - 1  # 일치: +1, 불일치: -1
#     # params.data *= mask
#     params.data[~mask] = d_align_[~mask]


def query(instruction):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res

BATCH = 64 
def batch_query(instruction):
    prompts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ins}\n\n### Response:\n" for ins in instruction]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t.split("### Response:")[1].strip() for t in texts]

# pred_lst = []
# for instruction in tqdm(instruction_lst):
#     pred = query(instruction)
#     pred_lst.append(pred)

pred_lst = []
for i in tqdm(range(0, len(instruction_lst), BATCH)):
    preds = batch_query(instruction_lst[i:i+BATCH])
    pred_lst.extend(preds)

output_lst = []
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    output_lst.append(input_data)

with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
