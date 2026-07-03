from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
from loss_func.repnoise_loss import rep_noise_loss
from transformers import Trainer
from transformers import logging
# from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)
from utils import prune_wanda_outlier,SupervisedDataset,prune_with_FI

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
# from transformers.models.falcon.modeling_falcon import FalconAttention
# from transformers.models.mistral.modeling_mistral import MistralAttention

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union


# Integrations must be imported before ML frameworks:
# ruff: isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
)

# ruff: isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
# from transformers.trainer_pt_utils import (
#     DistributedTensorGatherer,
#     EvalLoopContainer,
#     IterableDatasetShard,
#     LabelSmoother,
#     LayerWiseDummyOptimizer,
#     LengthGroupedSampler,
#     SequentialDistributedSampler,
#     distributed_broadcast_scalars,
#     distributed_concat,
#     find_batch_size,
#     get_model_param_count,
#     get_module_class_from_name,
#     get_parameter_names,
#     nested_concat,
#     nested_detach,
#     nested_numpify,
#     nested_xla_mesh_reduce,
#     reissue_pt_warnings,
#     remove_dummy_checkpoint,
#     set_rng_state_for_device,
# )
# from transformers.trainer_utils import (
#     PREFIX_CHECKPOINT_DIR,
#     BestRun,
#     EvalLoopOutput,
#     EvalPrediction,
#     HPSearchBackend,
#     HubStrategy,
#     PredictionOutput,
#     RemoveColumnsCollator,
#     SaveStrategy,
#     TrainerMemoryTracker,
#     TrainOutput,
#     check_target_module_exists,
#     default_compute_objective,
#     denumpify_detensorize,
#     enable_full_determinism,
#     find_executable_batch_size,
#     get_last_checkpoint,
#     has_length,
#     neftune_post_forward_hook,
#     number_of_arguments,
#     seed_worker,
#     set_seed,
#     speed_metrics,
# )
# from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    # is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    # is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)

from transformers.trainer import _is_peft_model


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


# if is_accelerate_available():
#     from accelerate import Accelerator, skip_first_batches
#     from accelerate import __version__ as accelerate_version
#     from accelerate.state import AcceleratorState
#     from accelerate.utils import (
#         AutocastKwargs,
#         DistributedDataParallelKwargs,
#         DistributedType,
#         TorchTensorParallelPlugin,
#         load_fsdp_model,
#         load_fsdp_optimizer,
#         save_fsdp_model,
#         save_fsdp_optimizer,
#     )

#     DATA_SAMPLERS = [RandomSampler]
#     if version.parse(accelerate_version) > version.parse("1.3.0"):
#         from accelerate.utils import TorchTensorParallelPlugin
#     if version.parse(accelerate_version) > version.parse("0.23.0"):
#         from accelerate.data_loader import SeedableRandomSampler

#         DATA_SAMPLERS += [SeedableRandomSampler]


import os
import logging
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from transformers import Trainer
from collections import defaultdict

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

import transformers.loss.loss_utils as lu

_orig_fixed_ce = lu.fixed_cross_entropy
def _patched_fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs):
    kwargs.pop("is_safe", None)

    if isinstance(num_items_in_batch, torch.Tensor) and num_items_in_batch.device != logits.device:
        num_items_in_batch = num_items_in_batch.to(logits.device)
    return _orig_fixed_ce(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)

lu.fixed_cross_entropy = _patched_fixed_cross_entropy

def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

def right_padding(sequences: list[torch.Tensor], padding_value: Number, max_length: int = 512) -> torch.Tensor:
    padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    if max_length is not None:
        current_length = padded.size(1)
        if current_length < max_length:
            # Need to pad further to reach max_length
            pad_width = max_length - current_length
            padded = torch.nn.functional.pad(padded, (0, pad_width), value=padding_value)
        elif current_length > max_length:
            # Truncate if longer than max_length
            padded = padded[:, :max_length]
    return padded

class AntidoteTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        loss = step()
        # with torch.no_grad():
        #     if self.round>=self.warm_up_round:
        #         for name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 param.grad *= self.mask[name]
        self.round+=1
        return loss.detach() / self.args.gradient_accumulation_steps

    def init(self, mask_ratio, sample_num):
        self.mask_ratio=mask_ratio
        self.round = 0
        self.args.sample_num=sample_num
        
        # self.warm_up_round = 11999
        
        
    def save_mask(self, save_path):
        self.model.model.seqlen = 2048
        if self.args.system_evaluate =="True":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        # self.mask = prune_with_FI(self.args, self,self.model.model, self.get_train_dataloader(), device=torch.device("cuda:0"))
        if self.args.sample_num==0:
            self.mask = prune_wanda_outlier(self.args, self.model.model, None, device=torch.device("cuda:0"))
        else:
            self.mask = prune_wanda_outlier(self.args, self.model.model, self.get_train_dataloader(), device=torch.device("cuda:0"))
        if self.args.system_evaluate =="True":
            end_event.record()
            torch.cuda.synchronize()
            ont_shot_time = start_event.elapsed_time(end_event)
            print("Estimated wanda time {} (h)".format(ont_shot_time/ 1000/3600))
            memory_usage = torch.cuda.memory_reserved()
            print(f"Wanda Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")
        torch.save(self.mask, save_path)



class LisaTrainer(Trainer):
    
    def get_alignment_dataloader(self,alignment_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))
    
    
    def init(self,  alignment_dataset):
        if self.args.alignment_step!=0 and self.args.guide_data_num>0:
            self.status = "alignment"
        else:
            self.status = "finetune"
        self.alignment_weights ={}
        self.finetune_weights ={}
        # self.gamma ={}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()
                # self.gamma[name]= torch.zeros_like(param)
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)
    
    def end_training(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.status == "alignment":
                    self.alignment_weights[name] = param.data.detach().clone()
                else:
                    self.finetune_weights[name] = param.data.detach().clone()
        
        
        
        
    
    def switch_model(self):
        sum_drift =0
        if self.status == "alignment":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name].to("cuda:0") - self.alignment_weights[name].to("cuda:0"))**2
            print("finetuning drift to consensus{}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name].to("cuda:0") - self.alignment_weights[name].to("cuda:0"))**2
            print("alignment drift to consensus{}".format(sum_drift))
        
        
        
    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch
    
    
    def check_mode(self, inputs):
        if self.status == "alignment":
            if self.clock% (self.args.alignment_step )  ==  0 and self.steps!=0 and self.args.finetune_step!=0:
                self.status ="finetune"
                self.switch_model()
                # print("swith from alignment to finetune {}".format(self.steps))
                self.clock=0
                
            else:
                # alignment need another input
                inputs = self.sample_from_alignment()
        else:
            if  self.clock% (  self.args.finetune_step  )  ==  0 and self.steps!=0 and self.args.alignment_step!=0 and self.args.guide_data_num>0:
                self.status ="alignment"
                self.switch_model()
                 # alignment need another input

                inputs = self.sample_from_alignment()
                # print("swith from finetune to alignment {}".format(self.steps))
                self.clock=0
        return inputs
            
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        # may change input due to mode change
        inputs = self.check_mode(inputs)
        model.train()
        
        inputs = self._prepare_inputs(inputs)
        
        
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.status =="alignment":
                # print("alignment_loss_prev: {}".format(loss.item()))
                if self.steps>0.1* len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        if param.requires_grad and self.args.rho>0:
                            # loss +=torch.sum(self.gamma[name] *  param)+ self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
                            loss += self.args.rho/2* torch.norm( param.to("cuda:0")- self.finetune_weights[name].to("cuda:0"))**2
                # print("alignment_loss: {}".format(loss.item()))
            else:
                # print("finetune_loss_prev: {}".format(loss.item()))
                
                if self.steps>0.1* len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        # we observe that for Gsm8k, proximal term will hurt convergence. Don't do proximal for the first few rounds.
                        if param.requires_grad and self.args.rho>0:
                            # loss += (- torch.sum(self.gamma[name] *  param )) + self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
                            loss +=  self.args.rho/2* torch.norm( param.to("cuda:0")- self.alignment_weights[name].to("cuda:0"))**2
                # print("finetune_loss: {}".format(loss.item()))
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        
        
        loss = step()    
        self.steps+=1
        self.clock+=1
        return loss.detach() / self.args.gradient_accumulation_steps




class LDIFSTrainer(Trainer):
    
    
    def init(self, model):
        import copy
       

        # Deep copy the object
        self.alignment_model = copy.deepcopy(model)

        # Ensure all tensors are in half precision
        # self.alignment_model = self.alignment_model.half()
        # self.alignment_model.eval()
        # Verifying if the parameters are in half precision
        # for param in model.parameters():
        #     print(param.dtype)  # Should print torch.float16 for all parameters
       
        self.steps = 0
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        import copy
        inputs = self._prepare_inputs(inputs)
        
        def step():
            def register_activation_hook(model):
                activations = {}
                hooks = []
                i=0
                for name, param in model.named_modules():
                    if name == f'base_model.model.model.layers.{i}.mlp':
                        param.name = name
                        def _hook(module, __, val):
                            activations[module.name] = val
                            # print(val)
                        hooks += [param.register_forward_hook(_hook)]
                        i+=1
                        # print(name)
                    
                return activations, hooks 
            
            activations, hooks = register_activation_hook(model)
    
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # if self.steps>=0* len(self.get_train_dataloader()) * self.args.num_train_epochs:
            # if self.steps>0.1* len(self.get_train_dataloader()) * self.args.num_train_epochs:
            
            def compare_models(model1, model2):
                for param1, param2 in zip(model1.parameters(), model2.parameters()):
                    if not torch.equal(param1, param2):
                        print("Mismatch found")
                        return False
                return True
            
            
            alignment_activations, alignment_model_hooks = register_activation_hook(self.alignment_model)            
            self.alignment_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])  
            
            # if compare_models(model, self.alignment_model):
            #     print("Models are identical")
            # else:
            #     print("Models differ")
            proximal_loss=0
            for name in alignment_activations:
                # print(alignment_activations[name])
                # print(alignment_activations[name].shape)
                
                # in some layers the proximal loss will be NAN, drop those overflow loss
                proximal_loss = self.args.rho/2* torch.norm( activations [name]- alignment_activations[name])**2
                if proximal_loss<0.1:
                    # print(name)
                    # print(proximal_loss)
                    loss += proximal_loss.to(loss.device)
            # print(loss)    
            # clean up before leaving
            for hook in hooks:
                hook.remove()
            hooks = []
            activations =  {}
            
            for hook in alignment_model_hooks:
                hook.remove()
            alignment_model_hooks = []
            alignment_activations = {}
    
            
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            return loss 
        loss = step()    
        
        self.steps+=1
        return loss.detach() / self.args.gradient_accumulation_steps
   
def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention) or isinstance(module, GemmaAttention) or isinstance(module, Qwen2Attention)or isinstance(module, Gemma2Attention):
        # if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention):
            module_list+= [module]
    # # print(module_list)
    return module_list
            

class SecurityVectorTrainer(Trainer):
    def init(self, harmless_dataset, base_model):
        self.regul_lambda = 0.01
        self.base_model = base_model
        self.harmless_dataloader = self.get_harmless_dataloader(harmless_dataset)
        self.harmless_data_iter = iter(self.harmless_dataloader)

    def get_harmless_dataloader(self,harmless_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import seed_worker
        from transformers.trainer_pt_utils import LengthGroupedSampler
        from torch.utils.data import DataLoader, RandomSampler

        data_collator = self.data_collator
  
        sampler = RandomSampler(harmless_dataset)

        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmless_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmless_dataset, **dataloader_params))

    def sample_from_harmless(self):
        # Get a  batch
        try:
            batch = next(self.harmless_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmless_data_iter = iter(self.harmless_dataloader)
            batch = next(self.harmless_data_iter)
        return batch

    def compute_kl_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes a teacher-student distillation loss where the base model logits
        act as the teacher on harmless batches.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = inputs.get("labels")

        outputs = model(**inputs)
        with torch.no_grad():
            base_outputs = self.base_model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        student_logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        teacher_logits = base_outputs["logits"] if isinstance(base_outputs, dict) else base_outputs.logits

        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels.ne(-100)
        else:
            valid_mask = torch.ones(
                shift_student_logits.shape[:-1],
                dtype=torch.bool,
                device=shift_student_logits.device,
            )

        student_log_probs = F.log_softmax(shift_student_logits, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        valid_mask = valid_mask.to(token_kl.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        reg_loss = (token_kl * valid_mask).sum() / denom

        loss = self.regul_lambda * reg_loss

        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs, skipping unstable batches.

        Args:
            model (`nn.Module`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.

        Returns:
            `torch.Tensor`: The training loss for this batch or a zero tensor if skipped.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmless_inputs = self.sample_from_harmless()
        harmless_inputs = self._prepare_inputs(harmless_inputs)
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            loss += self.compute_kl_loss(model, harmless_inputs)

        del inputs  # Free memory

        kwargs = {}

        # Use correct learning rate for LOMO optimizers
        # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        #     kwargs["learning_rate"] = self._get_learning_rate()

        # Handle multi-GPU training
        if self.args.n_gpu > 1:
            loss = loss.mean()  # Average loss across GPUs

        # Backward pass
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad.zero_()  #  Set gradient to zero

        return loss.detach() / self.args.gradient_accumulation_steps  # Continue normal training


class AsFTTrainer(Trainer):
    def init(self, regul_lambda, project_matrix):
        self.regul_lambda = regul_lambda
        self.project_matrix = project_matrix

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        reg_loss = torch.zeros_like(loss)
        regul_lambda = self.regul_lambda
        idx = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'lora_A' in name:
                lora_a = param
                lora_b_name = name.replace('lora_A', 'lora_B')
                lora_b = dict(model.named_parameters())[lora_b_name]
                delta_w = torch.mm(lora_b, lora_a)
                c_hat = self.project_matrix[idx].to(delta_w.device)
                identity = torch.eye(c_hat.shape[0], device=delta_w.device)
                # regularized_term = (identity - c_hat) @ delta_w
                regularized_term = delta_w - c_hat @ delta_w
                reg_loss += regul_lambda * torch.norm(regularized_term.to(reg_loss.device), p='fro') ** 2
                idx += 1

        # if self.model.training:
        #     wandb.log({"loss/reg_loss": reg_loss.item() / regul_lambda})
        loss = loss + reg_loss

        return (loss, outputs) if return_outputs else loss

class Panacea(BoosterAlignmentTrainer):
    def init(self, harmful_datast, model, tag):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0
        self.tag = tag
        self.eval_metric = []
        if self.tag == "eps" or self.tag == "gw" or self.tag == "log":
            self.epsilon = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.epsilon[name] = torch.zeros_like(param.data, requires_grad=False)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)
     
        def step_eps():

            # optimize g(w)
            with self.compute_loss_context_manager():
                loss_g =  self.compute_loss(model, inputs)
            if self.use_apex:
                with amp.scale_loss(loss_g, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss_g)
            stored_g_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            model.zero_grad()

            # optimize h(w)
            with self.compute_loss_context_manager():
                loss_h_origin =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss_h_origin, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss_h_origin)
            stored_h_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}

            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.epsilon[name] = param.grad.data
            epsilon_norm = self._grad_norm(self.epsilon)
            for name, param in model.named_parameters():
                if param.requires_grad:

                    self.epsilon[name] *= self.args.eps_rho / (epsilon_norm.to(self.epsilon[name].device) + 1e-7)
                    param.data += self.epsilon[name] # w + epsilon     perturb the weights with new epsilon              
                      
            model.zero_grad()

            # h(w + epsilon)
            with self.compute_loss_context_manager():
                loss_h =  self.compute_loss(model, harmful_inputs)
            clip_loss_h = torch.clamp(loss_h, max=5)
            if self.use_apex:
                with amp.scale_loss(clip_loss_h, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(clip_loss_h)

            # w
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data -= self.epsilon[name]

            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.grad.data = - (self.args.lamb * (param.grad.data - stored_h_grads[name]) - stored_g_grads[name])

            self.steps+=1
            if (self.steps - 1) % 500==0:
                self.statistic=epsilon_norm

            return loss_h

        loss = step_eps()
        return loss.detach() / self.args.gradient_accumulation_steps
    def get_epsilon(self):
        return self.epsilon

class SafeLoRA:
    def __init__(self, peft_model: torch.nn.Module, config, training_args):
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        self.model_ori = copy.deepcopy(peft_model)
        import transformers
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            load_in_8bit=False,
            cache_dir=training_args.cache_dir,
            device_map="cpu",
        )
        self.aligned_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            load_in_8bit=False,
            cache_dir=training_args.cache_dir,
            device_map="cpu",
        )
        

        self.project_matrix = self.get_aligned_matrix()
        if self.config.select_layers_type == 'threshold':
            self.model, _ = self.projected_weighted(self.project_matrix, self.config.threshold, show_info=True)
        elif self.config.select_layers_type == 'number':
            _, cos = self.projected_weighted(self.project_matrix, -1, show_info=False)
            thrs = np.sort(cos)[:self.config.num_proj_layers][-1]
            self.model, _ = self.projected_weighted(self.project_matrix, thrs, show_info=True)
        else:
            raise ValueError("The method of select_layer_type should be threshold or number.")


    def get_aligned_matrix(self):
        v = []
        proj_modules = list(self.peft_config.target_modules)
        for (b_name, b_param), (a_name, a_param) in zip(self.base_model.named_parameters(), self.aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules):
                vec = a_param.to(self.config.devices) - b_param.to(self.config.devices)
                vec = vec
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append(vec.detach().cpu())
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        idx = 0
        i = 0
        dis = []
        cos_total = []
        B = None
        for (name, param), (name_ori, param_ori) in zip(self.peft_model.named_parameters(), self.model_ori.named_parameters()):
            if 'lora' in name:
                # store the basis B when we encounter the LoRA B matrix (shape[0] == r)
                if param.shape[0] == self.peft_config.r:
                    B = copy.deepcopy(param_ori)
                    # continue to next param; B will be used for subsequent params
                    continue

                # For LoRA parameters that are not the B matrix, perform projection
                if B is None:
                    # if B has not been found yet, fallback to original parameter
                    param.data = param_ori
                    continue

                P = v[idx].to(param.device)
                # Project original weight into the aligned subspace
                W = torch.mm(P, param_ori.data)
                fW = torch.mm(W, B)
                ori = torch.mm(param_ori, B)
                W_new = torch.mm(P, param_ori.data)
                cos = float(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)).item())
                cos = float(np.round(cos, 5))
                cos_total.append(cos)

                if cos <= thrs_cos:
                    i += 1
                    param.data = W_new
                    if show_info:
                        print(f"Layer {name} is projected.")
                else:
                    param.data = param_ori

                dist = 1 / (1 + torch.norm(param.data.reshape(1, -1) - W.reshape(1, -1)))
                dis.append(dist.item())
                idx += 1

        if show_info:
            mean_dis = float(np.mean(dis)) if len(dis) > 0 else 0.0
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {mean_dis} (> 0.8 is better).")
        return self.peft_model, cos_total


class SafeLoRAConfig:
    def __init__(self, base_model_path, aligned_model_path, devices, select_layers_type='threshold', threshold=0.6, num_proj_layers=10):
        self.base_model_path = base_model_path 
        self.aligned_model_path = aligned_model_path  
        self.devices = devices
        self.select_layers_type = select_layers_type
        self.threshold = threshold
        self.num_proj_layers = num_proj_layers


