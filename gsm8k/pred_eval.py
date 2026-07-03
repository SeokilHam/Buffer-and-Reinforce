import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

access_token = next(open('../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--lora_folder3", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")
parser.add_argument("--neurons_path", default= "../user_neurons.json")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = ""
dataset = load_dataset("openai/gsm8k", 'main')
index=0
input_data_lst = []
for data in dataset["test"]:
    if  index<1000 :
        item = {}
        item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
        item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
        input_data_lst += [item]
        index+=1

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir,  padding_side="left", use_fast=True,token = access_token,model_max_length=512)
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto",   token = access_token )
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


if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder
    )
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

# model.eval()

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

model.eval()

# with open(args.neurons_path, "r", encoding="utf-8") as f:
#     diff_users = json.load(f)

# for name, module in model.named_modules():
#     if name in diff_users:
#         mask = torch.ones((module.in_features * module.out_features), device=module.weight.device)
#         mask[list(diff_users[name])] = 0
#         mask = mask.view(module.out_features, module.in_features)
#         module.weight.data *= mask

BATCH = 128 
def batch_query(instruction):
    prompts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ins['instruction']}\n\n### Response:\n" for ins in instruction]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t.split("### Response:")[1].strip() for t in texts]

def query(data):
    instruction = data["instruction"]
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


# pred_lst = []
# for data in tqdm(input_data_lst):
#     pred = query(data)
#     pred_lst.append(pred)

pred_lst = []
for i in tqdm(range(0, len(input_data_lst), BATCH)):
    preds = batch_query(input_data_lst[i:i+BATCH])
    pred_lst.extend(preds)

output_lst = []
correct = 0
total = 0

def extract_answer_number(sentence: str) -> float:
    import re
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

for input_data, pred in zip(input_data_lst, pred_lst):
    answer_ground_truth = extract_answer_number(input_data ["output"])
    answer = extract_answer_number(pred)
    input_data['output'] = pred
    # print(answer_ground_truth)
    
    if answer_ground_truth==answer:
        correct +=1 
        input_data["correct"] ="true"
    else:
        input_data["correct"] ="false"
    total += 1
    output_lst.append(input_data)
print("{:.2f}".format(correct/total*100))
output_lst .append("score={:.2f}".format(correct/total*100))
with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
