import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file

access_token = next(open('../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--lora_folder3", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = ""
dataset = load_dataset("gsm8k", 'main')
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


model.eval()


BATCH = 64 
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
