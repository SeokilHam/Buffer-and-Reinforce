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
parser.add_argument("--instruction_path", default='gcg')
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default= "../../cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []
if "gcg" in args.instruction_path:
    from datasets import load_dataset
    dataset =load_dataset("MatanBT/gcg-evaluated-data")
    index=0
    input_data_lst = []
    for example in dataset["llama3.1"]:
        if index<1000: 
            instance = {}
            instance["instruction"] = example["message_suffixed"]
            instruction_lst += [example["message_suffixed"]]
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
