#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import transformers
from transformers import TrainerCallback, get_cosine_schedule_with_warmup
# from torch.utils.data import Dataset
from datasets import Dataset
from trainer import LisaTrainer,LDIFSTrainer,AntidoteTrainer,AsFTTrainer,Panacea,SafeLoRA,SafeLoRAConfig,SecurityVectorTrainer
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict
from tqdm import tqdm
import json
import wandb
# wandb.init(mode="disabled")
wandb.init(mode="online")
sys.path.append('..')
import utils
import copy
from utils import SupervisedDataset
import re
# // Set access token (NB: Keep this private!)
access_token = next(open('huggingface_token.txt')).strip()


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# A_grad_norm = {
#     "q_proj": [0.0]*32,
#     "v_proj": [0.0]*32,
#     "gate_proj": [0.0]*32,
#     "up_proj": [0.0]*32,
#     "down_proj": [0.0]*32,
# }

# B_grad_norm = {
#     "q_proj": [0.0]*32,
#     "v_proj": [0.0]*32,
#     "gate_proj": [0.0]*32,
#     "up_proj": [0.0]*32,
#     "down_proj": [0.0]*32,
# }

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


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



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, is_safe, refusal_input_ids, refusal, token_length = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "is_safe", "refusal_input_ids", "refusal", "token_length"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        refusal_input_ids = torch.nn.utils.rnn.pad_sequence(
            refusal_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        if refusal_input_ids.shape[1] > input_ids.shape[1]:
            input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], refusal_input_ids.shape[1]-input_ids.shape[1]), self.tokenizer.pad_token_id)], dim=-1)
        else:
            refusal_input_ids = torch.cat([refusal_input_ids, torch.full((refusal_input_ids.shape[0], input_ids.shape[1]-refusal_input_ids.shape[1]), self.tokenizer.pad_token_id)], dim=-1)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        refusal = torch.nn.utils.rnn.pad_sequence(refusal, batch_first=True, padding_value=IGNORE_INDEX)
        if labels.shape[1] > refusal.shape[1]:
            refusal = torch.cat([refusal, torch.full((refusal.shape[0], labels.shape[1]-refusal.shape[1]), IGNORE_INDEX)], dim=-1)
        else:
            labels = torch.cat([labels, torch.full((labels.shape[0], refusal.shape[1]-labels.shape[1]), IGNORE_INDEX)], dim=-1)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            is_safe=is_safe,
            refusal_input_ids=refusal_input_ids,
            refusal_attention_mask=refusal_input_ids.ne(self.tokenizer.pad_token_id),
            refusal=refusal,
            token_length=token_length
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # if training_args.optimizer == "prune_afterfinetune" and training_args.no_harmful_dataset!= "True":
    #     train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=1,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
    #     print("harmful dataset")
    # else:
    print("finetuning dataset")
    if "BeaverTails_safe"  in data_args.data_path:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=data_args.poison_data_start)
    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=data_args.poison_data_start)
        # train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=1,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
    if "BeaverTails_safe" not in data_args.data_path:
        # For evaluate harmful testing loss
        # eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=5000, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        
        # For evaluate harmful training loss
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=100, benign_dataset=data_args.benign_dataset,poison_data_start=0)
    else:
        eval_dataset=SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=1,sample_num=5000, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        # eval_dataset = None 
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--lora_folder2", type=str, default="", help="Specify the lora path")
    parser.add_argument("--rho", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--poison_data_start", type=int, default=0, help="Specify the optimizer to use")
    parser.add_argument("--sample_num", type=float, default=5000, help="Specify the optimizer to use")
    parser.add_argument("--benign_dataset", type=str, default="data/sst2.json", help="Specify the optimizer to use")
    parser.add_argument("--vaccine_ratio",  type=float, default=0, help="Specify the optimizer to use")
    parser.add_argument("--lamb",  type=float, default=0.001, help="Specify the optimizer to use")
    parser.add_argument("--track_embedding_before_train",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--track_embedding_drift",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--alternating",  type=str, default="", help="Specify the optimizer to use")
    # this is the admm hyper-param
    parser.add_argument("--finetune_step",  type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--alignment_step",  type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--guide_data_num",  type=int, default=10000, help="Specify the optimizer to use")
    parser.add_argument("--dense_ratio",  type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--noise_variance",  type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--bad_sample_num",  type=float, default=1000, help="Specify the optimizer to use")
    parser.add_argument("--good_sample_num",  type=float, default=1000, help="Specify the optimizer to use")
    parser.add_argument("--system_evaluate",  type=str, default="True", help="Specify the optimizer to use")
    parser.add_argument("--no_harmful_dataset",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--no_safety_mask",  type=str, default="True", help="Specify the optimizer to use")
    parser.add_argument("--random_prune",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--full_model_prune",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--perturb_aware",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--alpha",  type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--meta_term",  type=str, default="True", help="Specify the optimizer to use")
    parser.add_argument("--full_finetuning",  type=str, default="False", help="Specify the optimizer to use")

    # Security Vector
    parser.add_argument("--regul_lambda",  type=float, default=1, help="Specify the optimizer to use")

    # Panacea
    parser.add_argument("--eps_rho",  type=float, default=1, help="Specify the optimizer to use")
    parser.add_argument("--tag",  type=str, default="", help="Specify the optimizer to use")
    parser.add_argument("--add_eps",  type=bool, default=True, help="Specify the optimizer to use")
    
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    # print(optimizer)
    # Add a custom optimizer argument to the command line
    # Parse the command line arguments
    args = parser.parse_args()
    random.seed(args.seed)

    # Set the seed for NumPy
    np.random.seed(args.seed)
    # Set the seed for PyTorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the optimizer choice in the training_args dataclass
    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.lamb = extra_args.lamb
    training_args.track_embedding_before_train = extra_args.track_embedding_before_train
    training_args.alternating = extra_args.alternating
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    data_args.bad_sample_num = extra_args.bad_sample_num
    data_args.good_sample_num = extra_args.good_sample_num
    data_args.poison_data_start = extra_args.poison_data_start
    training_args.guide_data_num = extra_args.guide_data_num
    training_args.rho = extra_args.rho
    training_args.finetune_step = extra_args.finetune_step
    training_args.alignment_step = extra_args.alignment_step
    training_args.dense_ratio = extra_args.dense_ratio
    training_args.noise_variance = extra_args.noise_variance
    training_args.model = model_args.model_name_or_path
    training_args.track_embedding_drift = extra_args.track_embedding_drift
    training_args.system_evaluate = args.system_evaluate
    training_args.no_harmful_dataset = extra_args.no_harmful_dataset
    training_args.no_safety_mask =extra_args.no_safety_mask
    training_args.random_prune=extra_args.random_prune
    training_args.full_model_prune=extra_args.full_model_prune
    training_args.sample_num = extra_args.sample_num
    training_args.alpha = extra_args.alpha
    training_args.meta_term = extra_args.meta_term
    training_args.model_max_length=512
    training_args.remove_unused_columns = False
    training_args.full_finetuning = extra_args.full_finetuning

    training_args.eps_rho = args.eps_rho
    training_args.tag = args.tag
    
    training_args.warmup_ratio = args.warmup_ratio
    # training_args.warmup_steps = args.warmup_steps

    training_args. perturb_aware = extra_args.perturb_aware
    
    if extra_args.optimizer== "LDIFS":
        # to prevent oom
        training_args.model_max_length=256
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        cache_dir=training_args.cache_dir,
        device_map="auto",
        token = access_token,
        # use_cache=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
        token = access_token

    )
    
    # Enable BF16 precision
    model = model.to(torch.bfloat16)
    
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
    # model = prepare_model_for_int8_training(model)
    if training_args.optimizer =="EWC" or  training_args.alternating =="single_lora":
        first_lora_trainable=True
        print("single_lora here !!!!!!!")
    else:
        first_lora_trainable=False

    base_model = copy.deepcopy(model)
    
    loar_alpha=64            
    if extra_args.lora_folder!="":
        print("Recover LoRA weights..")
        model = PeftModel.from_pretrained(
        model,
        extra_args.lora_folder,
        is_trainable=first_lora_trainable
        )
        # single lora method don't need to merge and load second lora
        
        if not first_lora_trainable:
            model = model.merge_and_unload()
            if extra_args.lora_folder2=="":
                # create new second lora for training 
                config = LoraConfig(
                    r=32,
                    lora_alpha=loar_alpha,
                    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    lora_dropout=0.1,
                    task_type="CAUSAL_LM",
                    )
                # initialize the model with the LoRA framework
                model = get_peft_model(model, config)    
            else:
                # load second lora and used for training
                model = PeftModel.from_pretrained(
                model,
                extra_args.lora_folder2,
                is_trainable=True
                )
                
                print(model.peft_config)  
    else:
        # create first lora
        print("Initialize Lora weights..")
        config = LoraConfig(
            r=32,
            lora_alpha=loar_alpha,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
            init_lora_weights=True
        )
        
        model = get_peft_model(model, config)
        
    model.train()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    elif training_args.optimizer == "lisa":
        trainer = LisaTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
        alignment_dataset  = SupervisedDataset(tokenizer=tokenizer, data_path="LAT_safe",sample_num=data_args.guide_data_num, poison_data_start=0, poison_ratio=1)
        trainer.init(alignment_dataset)
    elif training_args.optimizer == "panacea":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="LAT_harm", poison_ratio=1,sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=0)
        trainer = Panacea(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
        trainer.init(harmful_dataset, model, "eps")
    elif training_args.optimizer == "LDIFS":
        trainer = LDIFSTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(model)
    elif training_args.optimizer == "security_vector":
        harmless_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="LAT_safe", poison_ratio=0, sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=0)
        trainer = SecurityVectorTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
        trainer.init(harmless_dataset, base_model)
    elif training_args.optimizer == "asft":
        if training_args.system_evaluate =="True":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        class SafeLoRA:
            def __init__(self):
                super().__init__()
                
                self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B",
                    return_dict=True,
                    load_in_8bit=False,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    cache_dir=training_args.cache_dir,
                )
                self.aligned_model = transformers.AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    return_dict=True,
                    load_in_8bit=False,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    cache_dir=training_args.cache_dir,
                )
                self.project_matrix = self.get_aligned_matrix()

            def get_aligned_matrix(self):
                v = []
                proj_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
                for (b_name, b_param), (a_name, a_param) in zip(self.base_model.named_parameters(), self.aligned_model.named_parameters()):
                    if any(module in a_name for module in proj_modules):
                        vec = a_param - b_param
                        vec = vec.to("cuda")
                        vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                        v.append(vec.detach().cpu())
                return v
        
        safelora = SafeLoRA()
        project_matrix = safelora.get_aligned_matrix()
        if  training_args.system_evaluate =="True":
            end_event.record()
            torch.cuda.synchronize()
            one_shot_time = start_event.elapsed_time(end_event)
            print("Estimated one shot time {} (h)".format(one_shot_time/ 1000/3600))
            memory_usage = torch.cuda.memory_reserved()
            print(f"Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")
        trainer = AsFTTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
        trainer.init(args.regul_lambda, project_matrix)
    elif training_args.optimizer == "SafeInstruct":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="LAT_safe", poison_ratio=1,sample_num=int(data_args.sample_num*0.1),poison_data_start=5000)
        data_module["train_dataset"] = data_module["train_dataset"] + harmful_dataset
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
    elif training_args.optimizer == "antidote":
        trainer = AntidoteTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(training_args.dense_ratio, data_args.sample_num)
    else:
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) # , optimizers=(optimizer, lr_scheduler)
   
        
    # calcualte the training steps to calculate gpu time
    num_train_samples = len(data_module["train_dataset"])
    num_train_epochs = training_args.num_train_epochs
    train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * (num_train_samples // effective_batch_size)
    print(total_steps)
    class GPUTimeCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic = 0
            self.total_time = 0
            self.record_time = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_event = torch.cuda.Event(enable_timing=True)
            state.end_event = torch.cuda.Event(enable_timing=True)
            state.start_event.record()
    

        def on_step_end(self, args, state, control, **kwargs):
            state.end_event.record()
            torch.cuda.synchronize()
            step_time = state.start_event.elapsed_time(state.end_event)
            self.average_statistic =  (self.average_statistic* self.record_time +step_time) / (self.record_time+1)  
            self.total_time += step_time
            self.record_time +=1
            if self.record_time%1==0:
                print("Estimated total time {} (h)".format(self.average_statistic*total_steps/ 1000/3600))
                wandb.log({"Cost/total time (h)": self.total_time/1000/3600})

        
    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic_memory = 0
            self.record_time_memory = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_memory = torch.cuda.memory_reserved()
            # print(self.record_time_memory)
            
        def on_step_end(self, args, state, control, **kwargs):
            total_memory = 0
            max_memory = 0
            num_gpus = torch.cuda.device_count()
            
            for i in range(num_gpus):
                total_memory += torch.cuda.memory_reserved(i)
                max_memory = max(max_memory, torch.cuda.max_memory_allocated(i))
                
            state.end_memory = total_memory

            self.average_statistic_memory =  (self.average_statistic_memory* self.record_time_memory +state.end_memory ) / (self.record_time_memory+1)  
            self.record_time_memory +=1
            if self.record_time_memory%1==0:
                print(f"Step {state.global_step}: {max_memory / (1024 ** 3):.2f} GB GPU memory used")
                wandb.log({"Cost/Average Memory per step (GB)": self.average_statistic_memory / (1024 ** 3)})
                wandb.log({"Cost/Max Memory per step (GB)": max_memory / (1024 ** 3)})

                
    
    if training_args.system_evaluate =="True":
        trainer.add_callback(GPUTimeCallback())
        trainer.add_callback(GPUMemoryCallback())
        # trainer.add_callback(EmbeddingCallback())
    
    class evaluationCallback(TrainerCallback):
        # every eval_steps output the gradient norm 
        
        def __init__(self):
            super().__init__()
            self.step=0
            
        def query(self, instruction, model ):
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
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            s = generation_output[0]
            output = tokenizer.decode(s, skip_special_tokens=True)
            res = output.split("### Response:")[1].strip()
            return res


        def on_step_end(self, args, state, control, model , **kwargs):
            self.step+=1
            if self.step%args.eval_steps==0:
                model.eval()
                instruction_lst = []
                from datasets import load_dataset
                dataset =load_dataset("PKU-Alignment/BeaverTails")
                index=0
                input_data_lst = []
                for example in dataset["30k_test"]:
                    if  index<200 and not example["is_safe"]:
                # for example in dataset["30k_train"]:
                #     if  index<100 and  example["is_safe"]:
                        instance = {}
                        instance["instruction"] = example["prompt"]
                        instruction_lst += [example["prompt"]]
                        input_data_lst += [instance]
                        index+=1
                
                pred_lst = []
                for instruction in tqdm(instruction_lst):
                    pred = self.query(instruction, model )
                    pred_lst.append(pred)

                output_lst = []
                for input_data, pred in zip(input_data_lst, pred_lst):
                    input_data['output'] = pred
                    output_lst.append(input_data)
                if "smooth" in extra_args.lora_folder:
                    file_name = "smooth_harmful_score_steps_{}_{}".format(data_args.poison_ratio, self.step )
                else:
                    file_name = "sft_harmful_score_steps_{}_{}".format(data_args.poison_ratio, self.step )
                with open(file_name, 'w') as f:
                    json.dump(output_lst, f, indent=4)
            
            
    
    # track the embedding before train
    if training_args.track_embedding_before_train=="True":
        from utils import track_embedding
        track_embedding(extra_args, trainer.get_eval_dataloader(), model)
        
    
    # if training_args.optimizer == "finetune_undercover":
    # trainer.add_callback(evaluationCallback())
    
    if training_args.num_train_epochs>0:
        trainer.train()

    if training_args.optimizer == "admm":
        trainer.end_training()
    
    # perturb the model
   
    if training_args.optimizer == "antidote" and training_args.random_prune!="True":
        trainer.save_mask(training_args.output_dir+ "/bad_mask.pt")    
    # trainer.save_model(output_dir=training_args.output_dir)
    if  training_args.optimizer == "antidote":
        if training_args.system_evaluate =="True":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if training_args.random_prune=="True":
            for name, param in model.named_parameters():
                # name= name[:-7]
                if param.requires_grad:
                    shape = param.shape
                    total_elements = param.numel()
                    num_non_zero_elements = int(total_elements * training_args.dense_ratio)
                    mask = torch.zeros(total_elements)
                    non_zero_indices = torch.randperm(total_elements)[:num_non_zero_elements]
                    mask[non_zero_indices] = 1
                    mask = mask.view(shape).to("cuda:0")
                    param.data *= (1-mask)
        else:
            bad_mask = torch.load(training_args.output_dir+"/bad_mask.pt")
            
            # print(bad_mask)
            # new_bad_mask = {}
            # for name in bad_mask:
            #     new_name = name.split('.', 1)[1]
            #     print (new_name)
            #     new_bad_mask[new_name] = bad_mask [name] 
            # bad_mask = new_bad_mask
            # torch.save(bad_mask, training_args.output_dir+"/bad_mask.pt")
            
            
            for name, param in model.named_parameters():
                # name= name[:-7]
                # print(name)
                # print("hi")
                if name in bad_mask:
                    param.data *= (1-bad_mask[name])
    
        if  training_args.system_evaluate =="True":
            end_event.record()
            torch.cuda.synchronize()
            ont_shot_time = start_event.elapsed_time(end_event)
            print("Estimated one shot time {} (h)".format(ont_shot_time/ 1000/3600))
            memory_usage = torch.cuda.memory_reserved()
            print(f"Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")
            
    # calculate the embedding drift after train
    if training_args.track_embedding_drift=="True":
        from utils import calculate_drift2first_embedding
        calculate_drift2first_embedding(extra_args, trainer.get_eval_dataloader(),model)
        
    if training_args.optimizer == "panacea":
        if training_args.system_evaluate =="True":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        print("add eps:",args.add_eps)
        print("tag:",args.tag)
        if args.tag == "eps" and args.add_eps == True:
            print("start adding eps")
            epsilon = trainer.get_epsilon()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data += epsilon[name]

        if  training_args.system_evaluate =="True":
            end_event.record()
            torch.cuda.synchronize()
            ont_shot_time = start_event.elapsed_time(end_event)
            print("Estimated one shot time {} (h)".format(ont_shot_time/ 1000/3600))
            memory_usage = torch.cuda.memory_reserved()
            print(f"Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")

    
    trainer.save_state()
    if training_args.optimizer != "safelora":
        model.save_pretrained(training_args.output_dir)
    else:
        if training_args.system_evaluate =="True":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        safe_config = SafeLoRAConfig(
            # base_model_path='meta-llama/Meta-Llama-3-8B',
            # aligned_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
            base_model_path='Qwen/Qwen3-4B-Base',
            aligned_model_path='Qwen/Qwen3-4B-Instruct-2507',
            # base_model_path='google/gemma-2-9b',
            # aligned_model_path='google/gemma-2-9b-it',
            devices='cuda' if torch.cuda.is_available() else 'cpu'
        )
        safelora = SafeLoRA(model, safe_config, training_args)

        if  training_args.system_evaluate =="True":
            end_event.record()
            torch.cuda.synchronize()
            ont_shot_time = start_event.elapsed_time(end_event)
            print("Estimated one shot time {} (h)".format(ont_shot_time/ 1000/3600))
            memory_usage = torch.cuda.memory_reserved()
            print(f"Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")

        projected_model = safelora.model
        projected_model.save_pretrained(training_args.output_dir)
        

if __name__ == "__main__":
    train()
