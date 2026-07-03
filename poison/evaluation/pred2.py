import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file

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


print("Starting Hugging Face Weight Injection...")
layers = model.model.layers  # LlamaForCausalLM 등의 표준 경로
weight_list = torch.load(os.path.join(args.model_folder[:-15], 'lora_ABC.pt'), map_location=torch.device('cpu'))

for i, layer in enumerate(layers):
    # 키 이름 생성 (저장할 때의 네이밍 규칙에 따름)
    # 예: q_proj_0weight, gate_proj_0weight 등
    suffix = f"_{i}weight"
    
    # -----------------------------------------------------------
    # 1. Self Attention (Q, V) 업데이트
    # -----------------------------------------------------------
    # Q Projection
    if ('q_proj' + suffix) in weight_list:
        new_weight = weight_list['q_proj' + suffix]
        layer.self_attn.q_proj.weight.data.copy_(
            new_weight.to(device=layer.self_attn.q_proj.weight.device, dtype=layer.self_attn.q_proj.weight.dtype)
        )
        print(f"[{i}] Updated q_proj")

    # V Projection
    if ('v_proj' + suffix) in weight_list:
        new_weight = weight_list['v_proj' + suffix]
        layer.self_attn.v_proj.weight.data.copy_(
            new_weight.to(device=layer.self_attn.v_proj.weight.device, dtype=layer.self_attn.v_proj.weight.dtype)
        )
        print(f"[{i}] Updated v_proj")

    # (참고) K Projection은 보통 SALoRA에서 건드리지 않으므로 패스합니다.

    # -----------------------------------------------------------
    # 2. MLP (Gate, Up, Down) 업데이트
    # -----------------------------------------------------------
    # Gate Projection
    if ('gate_proj' + suffix) in weight_list:
        new_weight = weight_list['gate_proj' + suffix]
        # HF 모델에서 MLP는 보통 layer.mlp 안에 있습니다.
        layer.mlp.gate_proj.weight.data.copy_(
            new_weight.to(device=layer.mlp.gate_proj.weight.device, dtype=layer.mlp.gate_proj.weight.dtype)
        )
        print(f"[{i}] Updated gate_proj")

    # Up Projection
    if ('up_proj' + suffix) in weight_list:
        new_weight = weight_list['up_proj' + suffix]
        layer.mlp.up_proj.weight.data.copy_(
            new_weight.to(device=layer.mlp.up_proj.weight.device, dtype=layer.mlp.up_proj.weight.dtype)
        )
        print(f"[{i}] Updated up_proj")

    # Down Projection
    if ('down_proj' + suffix) in weight_list:
        new_weight = weight_list['down_proj' + suffix]
        layer.mlp.down_proj.weight.data.copy_(
            new_weight.to(device=layer.mlp.down_proj.weight.device, dtype=layer.mlp.down_proj.weight.dtype)
        )
        print(f"[{i}] Updated down_proj")

print("✅ All weights injection completed.")


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
