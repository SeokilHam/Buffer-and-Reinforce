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


class VlguardTrainer(Trainer):
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
            "batch_size": 4,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))
    
    
    def init(self,  alignment_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_datast)
            self.alignment_data_iter = iter(self.alignment_dataloader)
            
    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.alignment_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.alignment_data_iter = iter(self.alignment_dataloader)
            batch = next(self.alignment_data_iter)
        return batch

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        alignment_inputs = self.sample_from_alignment()
        alignment_inputs = self._prepare_inputs(alignment_inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) + self.args.lamb* self.compute_loss(model, alignment_inputs)
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
        return loss.detach() / self.args.gradient_accumulation_steps


class BoosterAlignmentTrainer(Trainer):

    def get_harmful_dataloader(self,harmful_datast) -> DataLoader:
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
  
        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))
    
    
    def init(self,  harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0


    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch



    
    
    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        
        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.sam_state["gradient"][module] = e_r.detach().clone()
   
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    #original sam 
                    ( poison_grads_representation[name] ).norm(p=2).to("cuda:0")
                    #asam 
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            # first backward gradient for harmful dataset    
            with self.compute_loss_context_manager():
                loss =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")            
            stored_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            model.zero_grad()
            
            # Take step with the harmful perturbation
            with torch.no_grad():
                grad_norm = self._grad_norm(stored_grads)+ 1e-7
            # perturb the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data += self.args.rho*stored_grads[name]/grad_norm
                    param.data -= self.args.alpha*stored_grads[name].to(param.device)/grad_norm.to(param.device)
          
            # backward the gradient after harmful perturbation
            with self.compute_loss_context_manager():
                loss2 =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            perturb_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
            
            
            model.zero_grad()
            
            # recover the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += self.args.alpha*stored_grads[name].to(param.device)/grad_norm.to(param.device)
              
            # Vaccine+Booster here
            if self.args.perturb_aware =="True":
                self.sam_state = {}
                self.sam_state ["hooks"] = []
                self.sam_state ["gradient"] = {}
                # do forward backward on safety data
                self.pre_first_step(model)
                # first backward
                loss4 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss4, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss4)
                self.after_first_step(model)
                model.zero_grad()
                self.pre_second_step(model)
                loss3 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                # cancel the perturbation
                self.after_second_step(model)
                # sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # param.grad.data=param.grad.data - (self.args.alpha +self.args.lamb/self.args.rho)*stored_grads[name] +self.args.lamb/self.args.rho* perturb_grads[name]
                        if self.args.meta_term=="False":
                            param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name].to(param.device) 
                        else:
                            param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name].to(param.device) -self.args.lamb* perturb_grads[name].to(param.device)
                        
                self.steps+=1
                if self.steps%1000==0:
                    self.statistic=0
                    self.statistic += sum([torch.norm(stored_grads[name])**2 for name, param in model.named_parameters() if param.requires_grad ]).detach()
                    print("harmful gradient norm {}".format(self.statistic),flush=True)
                    print("harmful loss {}".format(loss),flush=True)
                return loss3
            # Plain Booster here
            else:
            # else:
            # Finally backward for minimize safety gradient
            # print(loss)
                # calculate the alignment grad
                with self.compute_loss_context_manager():
                    loss3 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                    
                # Finally, sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if self.args.meta_term=="False":
                            # print("haha",flush=True)
                            param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name].to(param.device) 
                        else:
                            param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name].to(param.device) -self.args.lamb* perturb_grads[name].to(param.device)
        
                    
                self.steps+=1
                if self.steps%1==0 :
                    self.statistic=0
                    self.statistic += grad_norm.detach()
                    # self.statistic += loss-loss2
                    print("harmful gradient norm {}".format(self.statistic),flush=True)
                    print("loss change {}".format(loss-loss2),flush=True)
                    print("harmful loss {}".format(loss),flush=True)
            return loss3
        
        loss = step()   
        return loss.detach() / self.args.gradient_accumulation_steps



class ADMMTrainer(Trainer):
    
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
            
            
class BaseTrainer(Trainer):
    def init(self, mask_ratio):
        self.mask_ratio=mask_ratio
        self.round = 0
        # self.warm_up_round = 11999
        
        
    def save_mask(self, save_path):
        # # OWL here!!!!!!!
        self.model.model.seqlen = 2048
        self.mask = prune_wanda_outlier(self.args, self.model.model, self.get_train_dataloader(), device=torch.device("cuda:0"), prune_n=0, prune_m=0)
        torch.save(self.mask, save_path)
        
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
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

        # if isinstance(self.optimizer,ESAM ):
        # print("calling sam")
        self.sam_state = {}
        self.sam_state ["hooks"] = []
        self.sam_state ["gradient"] = {}
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    
    
    
    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        
        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale.to(grad.device)
            self.sam_state["gradient"][module] = e_r.detach().clone()
   
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    #original sam 
                    ( poison_grads_representation[name] ).norm(p=2).to("cuda:0")
                    #asam 
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm


        
class RepNoiseTrainer(Trainer):
    def init(self,  harmful_dataset):
        # reploss needs standard dataset, load alpaca here
        from transformers.trainer_utils import ( seed_worker)
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        self.harmful_dataloader = self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))
        
        
    
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Get an iterator from the DataLoader
        data_iter = iter(self.harmful_dataloader)
        # Get the next batch
        harmful_inputs = next(data_iter)
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                # loss = self.compute_loss(model, inputs)
                loss = rep_noise_loss(model,harmful_inputs,inputs, beta = self.args.lamb, alpha = self.args.rho)
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

        return loss.detach() / self.args.gradient_accumulation_steps

class LIMATrainer(Trainer):
    def init(self, ref_model, beta_max, regul_lambda):
        self.ref_model = ref_model
        self.harmful_correct = 0
        self.harmless_correct = 0
        self.num_harmful = 0
        self.num_harmless = 0
        self.beta_max = beta_max
        self.regul_lambda = regul_lambda
        
        target_strs = ["as","As","sorry","Sorry","I"]
        self.target_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in target_strs]

    # def get_loss_n_score(self, outputs, labels):
    #     logits = outputs.logits[:, :-1, :]           # [B, T-1, V]
    #     targets = labels[:, 1:]
    #     loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none', ignore_index=-100).reshape(logits.shape[0], -1)   # (B, T)
    #     mask = (targets != -100).float()      # 답변 구간 마스크
    #     denom = mask.sum(1).clamp_min(1.0)   # 각 시퀀스의 유효 토큰 수
    #     # loglikelihood = -(loss * mask).sum(1) / denom
    #     # confidence = torch.exp(loglikelihood/20)
    #     confidence = -(loss * mask).sum(1) / denom
    #     return loss, confidence.detach()

    def get_loss(self, outputs, labels):
        logits = outputs.logits[:, :-1, :]           # [B, T-1, V]
        targets = labels[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none', ignore_index=-100).reshape(logits.shape[0], -1)   # (B, T)
        return loss

    def get_sorry_score(self, outputs, prompt_len):
        softmax_logits = outputs.logits.softmax(-1)
        confidence = []
        for i in range(len(prompt_len)):
            conf_sum = 0
            max_val = 0
            for idx in self.target_ids:
            #     if max_val < softmax_logits[i, prompt_len[i]-1, idx]:
            #         max_val = softmax_logits[i, prompt_len[i]-1, idx]
            # confidence.append(max_val[0])
                conf_sum += softmax_logits[i, prompt_len[i]-1, idx]
            confidence.append(conf_sum[0])
        confidence = torch.stack(confidence, dim=0)
        return (confidence)
    
    def get_label_score(self, outputs, labels, prompt_len):
        softmax_logits = outputs.logits.softmax(-1)
        confidence = []
        for i in range(len(prompt_len)):
            conf_sum = softmax_logits[i, prompt_len[i]-1, labels[i][prompt_len[i]]]
            confidence.append(conf_sum)
        confidence = torch.stack(confidence, dim=0)
        return (confidence)
    
    def seq_logprob(self, outputs, labels):
        """
        반환: [B] (평균 토큰 로그확률)
        labels에서 프롬프트/패딩은 -100이어야 함.
        """
        logits = outputs.logits  # [B, T, V]
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)       # shift
        tgt  = labels[:, 1:]                                  # [B, T-1]
        valid = (tgt != -100).to(logp.dtype)                  # [B, T-1]
        # 타겟 토큰의 로그확률 수집
        token_logp = logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid
        lengths = valid.sum(dim=1).clamp_min(1)
        return token_logp.sum(dim=1) / lengths
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        refusal_input_ids = inputs.pop("refusal_input_ids") if "refusal_input_ids" in inputs else None
        refusal_labels = inputs.pop("refusal") if "refusal" in inputs else None
        refusal_attention_mask = inputs.pop("refusal_attention_mask") if "refusal_attention_mask" in inputs else None    
        prompt_len = inputs.pop("token_length") if "token_length" in inputs else None
        # if refusal_labels is not None:
        #     # with torch.no_grad(): 
        #     #     ref_outputs = self.ref_model(**inputs)
        #     #     # ref_refusal_outputs = self.ref_model(input_ids=refusal_input_ids, attention_mask=refusal_attention_mask)
        #     #     ref_refusal_confidence = self.get_sorry_score(ref_outputs, prompt_len)
        #     #     ref_label_confidence = self.get_label_score(ref_outputs, labels, prompt_len)
        #     refusal_outputs = model(input_ids=refusal_input_ids, attention_mask=refusal_attention_mask)
        #     refusal_loss = self.get_loss(refusal_outputs, refusal_labels) # 필요한가? ORPO로만 학습해도 될거 같은데?
        #     # refusal_confidence = self.get_sorry_score(refusal_outputs, prompt_len)
        outputs = model(**inputs)
        loss = self.get_loss(outputs, labels)
        # label_confidence = self.get_label_score(outputs, labels, prompt_len)
        refusal_confidence = self.get_sorry_score(outputs, prompt_len)
        # alpha = (label_confidence.exp() - refusal_confidence.exp()).sigmoid()* 2 - 1
        # import pdb; pdb.set_trace()
        # alpha = (label_confidence - refusal_confidence).sigmoid().detach()
        alpha = refusal_confidence.detach()
        # loss = alpha * loss.mean(-1)
        # y_mask = (alpha > 0.).float()
        # refusal_mask = (alpha <= 0.).float()
        # y_mask = (alpha > 0.5).float().detach() 
        # refusal_mask = (alpha <= 0.5).float().detach() #* (alpha)
        y_mask = (alpha < 0.5).float().detach() 
        refusal_mask = (alpha >= 0.5).float().detach() #* (alpha)
        # y_mask = alpha
        # refusal_mask = 1-alpha
        # loss_y = y_mask * (1 - refusal_confidence.exp()) * loss.mean(-1) 
        # loss_r = refusal_mask.detach() * (-alpha) * refusal_loss.mean(-1)
        # loss_y = y_mask * alpha * loss.mean(-1) 
        # loss_r = refusal_mask.detach() * (alpha) * refusal_loss.mean(-1)
        loss_y = y_mask * (1-alpha) * loss.mean(-1) 
        # loss_r = refusal_mask.detach() * (alpha) * refusal_loss.mean(-1)

        # margin = (label_confidence - refusal_confidence) - (ref_label_confidence - ref_refusal_confidence)
        # beta = torch.clamp(alpha * self.beta_max, min=0.02, max=self.beta_max) * (2*(alpha > 0.5).float()-1).detach()
        # loss_dpo = F.softplus(-beta.detach() * margin.clamp(-20, 20)) 
        eps = 1e-12
        margin = (refusal_confidence+eps).log() - (1-refusal_confidence+eps).log()
        label_nll = self.seq_logprob(outputs, labels)
        # refusal_nll = (refusal_confidence+eps).log() - (1-refusal_confidence+eps).log() 
        # safe_margin = label_nll - refusal_nll * refusal_confidence
        # unsafe_margin = refusal_nll.clamp(min=eps, max=1.0 - eps) # - self.seq_logprob(refusal_outputs, refusal_labels)
        # margin = safe_margin * y_mask + unsafe_margin * refusal_mask
        # beta = alpha * (2*(alpha > 0.5).float()-1).detach() * self.beta_max
        # beta = self.beta_max * (1-alpha) * (2*(alpha < 0.5).float()-1).detach()
        # loss_dpo = F.softplus(-beta * margin.clamp(-8, 8)) 
        loss_dpo = alpha * F.softplus(-self.beta_max * margin.clamp(-8, 8)) #+ (1-alpha) * F.softplus(self.beta_max * margin.clamp(-8, 8))
        
        # if self.state.global_step == 975:
        #     import pdb; pdb.set_trace()
        # if torch.isnan(loss_y).any() or torch.isnan(loss_r).any() or torch.isnan(loss_dpo).any():
        #     import pdb; pdb.set_trace()
        # batch_size, _, vocab = refusal_outputs["logits"].shape
        # student_soft = F.log_softmax(refusal_outputs["logits"].view(-1, vocab) / 1, dim=-1)
        # teacher_soft = F.softmax(ref_refusal_outputs["logits"].view(-1, vocab) / 1, dim=-1)
        # refusal_distill_loss = F.kl_div(student_soft, teacher_soft, reduction='none').sum(-1).reshape(batch_size, -1)
        # # adaptive_coef = refusal_loss.mean(-1).exp()/(refusal_distill_loss.mean(-1).exp() + refusal_loss.mean(-1).exp())
        # loss_r = refusal_mask * (alpha * refusal_loss.mean(-1) + (1-alpha) * refusal_distill_loss.mean(-1))
        # loss_dpo = torch.zeros_like(loss_r)
        # loss = y_mask * loss.mean(-1) - refusal_mask**2 * loss.mean(-1)
        # print("is_safe:", inputs.get("is_safe", None))
        # # print("train:", label_confidence.exp(), refusal_confidence.exp())
        # # print("ref:", ref_label_confidence.exp(), ref_refusal_confidence.exp())
        # # print("train/ref:", label_confidence/ (ref_label_confidence+1e-7), ref_refusal_confidence/(refusal_confidence+1e-7))
        # print("alpha:", alpha)
        # print("y_mask:", y_mask, "refusal_mask:", refusal_mask)
        # # print("adaptive_coef: ", adaptive_coef)
        # # print("fixed_loss: ", alpha * refusal_loss.mean(-1), "distill_loss: ", (1-alpha) * refusal_distill_loss.mean(-1))
        # # print("beta: ", beta)
        # # print("margin: ", margin)
        # print("loss_y: ", loss_y)
        # # print("loss_r: ", loss_r)
        # print("loss_dpo: ", loss_dpo)
        # import pdb; pdb.set_trace()
        loss = (loss_y + self.regul_lambda*loss_dpo).mean()
        
        is_safe = inputs.pop("is_safe") if "is_safe" in inputs else None
        if is_safe is not None:
            y_gt_mask = torch.tensor(is_safe).to(alpha.device).float()
            refusal_gt_mask = 1 - torch.tensor(is_safe).to(alpha.device).float()
            self.num_harmful += torch.tensor(is_safe).to(alpha.device).float().sum()
            self.num_harmless += (1-torch.tensor(is_safe).to(alpha.device).float()).sum()
            self.harmful_correct += ((alpha < 0.5) == torch.tensor(is_safe).to(alpha.device)).float().sum()
            self.harmless_correct += ((alpha > 0.5) == (~torch.tensor(is_safe).to(alpha.device))).float().sum()
            harmful_accuracy = self.harmful_correct / (self.num_harmful + self.num_harmless)
            harmless_accuracy = self.harmless_correct / (self.num_harmless + self.num_harmful)
            # accuracy = ((alpha > 0) == torch.tensor(is_safe).to(alpha.device)).float().mean()

        if self.model.training:
            wandb.log({"loss/L_y": (y_gt_mask * loss.mean(-1)).mean().item(), "loss/regul_safe": ((1-alpha) * F.softplus(self.beta_max * margin.clamp(-8, 8))).mean().item(), "loss/regul_unsafe": (alpha * F.softplus(-self.beta_max * margin.clamp(-8, 8))).mean().item()})
            wandb.log({"confidence/train_label": label_nll[y_gt_mask.bool()].exp().mean().item(), "confidence/train_refusal": refusal_confidence[refusal_gt_mask.bool()].mean().item()})
            wandb.log({"accuracy/harmful_accuracy": harmful_accuracy.item(), "accuracy/harmless_accuracy": harmless_accuracy.item()})

        return (loss, outputs) if return_outputs else loss

class BufferLoRATrainer(Trainer):
    # def init(self, harmless_dataset):
    #     self.harmless_dataloader = self.get_harmless_dataloader(harmless_dataset)
    #     self.harmless_data_iter = iter(self.harmless_dataloader)
    
    def init(self, regul_lambda, bufferlora_dict=None):
        self.regul_lambda = regul_lambda
        self.bufferlora_dict = bufferlora_dict

    # def get_harmless_dataloader(self,harmless_dataset) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].

    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    #     training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
     
    #     from transformers.trainer_utils import seed_worker
    #     from transformers.trainer_pt_utils import LengthGroupedSampler
    #     from torch.utils.data import DataLoader, RandomSampler

    #     data_collator = self.data_collator
  
    #     sampler = RandomSampler(harmless_dataset)

    #     dataloader_params = {
    #         "batch_size": self.args.per_device_train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #     }

    #     if not isinstance(harmless_dataset, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = sampler
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker

    #     return self.accelerator.prepare(DataLoader(harmless_dataset, **dataloader_params))

    # def sample_from_harmless(self):
    #     # Get a  batch
    #     try:
    #         batch = next(self.harmless_data_iter)
    #     except (StopIteration):
    #         # If the iterator is exhausted, create a new iterator
    #         self.harmless_data_iter = iter(self.harmless_dataloader)
    #         batch = next(self.harmless_data_iter)
    #     return batch

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
        # for i, (name, param) in enumerate(model.named_parameters()):
        #     if 'lora_A' in name:
        #         # layer_idx = int(re.findall(r'\d+', name)[0])
        #         # if layer_idx < 7 or layer_idx > 12:
        #         #     continue
        #         lora_a = param
        #         lora_b_name = name.replace('lora_A', 'lora_B')
        #         lora_b = dict(model.named_parameters())[lora_b_name]
        #         delta_w = torch.mm(lora_b, lora_a)
        #         # model_weight_name = name.replace('.lora_A.default', '')
        #         # model_weight = dict(model.named_parameters())[model_weight_name].clone().detach().to(delta_w.device)
        #         # regularized_term = F.cosine_similarity(model_weight.view(-1), delta_w.view(-1), dim=-1)
        #         buffer_lora_a = self.bufferlora_dict[name.replace('.default', '')].clone().detach().to(delta_w.device)
        #         buffer_lora_b = self.bufferlora_dict[lora_b_name.replace('.default', '')].clone().detach().to(delta_w.device)
        #         buffer_delta_w = torch.mm(buffer_lora_b, buffer_lora_a)
        #         regularized_term = F.cosine_similarity(buffer_delta_w.view(-1), delta_w.view(-1), dim=-1)
        #         reg_loss += regul_lambda * torch.norm(regularized_term, p='fro').to(reg_loss.device) ** 2
        #         idx += 1
        # if self.model.training:
        #     wandb.log({"loss/reg_loss": reg_loss.item() / regul_lambda})
        loss = loss + reg_loss

        return (loss, outputs) if return_outputs else loss
    
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     """
    #     Perform a training step on a batch of inputs, skipping unstable batches.

    #     Args:
    #         model (`nn.Module`): The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.

    #     Returns:
    #         `torch.Tensor`: The training loss for this batch or a zero tensor if skipped.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     harmless_inputs = self.sample_from_harmless()
    #     harmless_inputs = self._prepare_inputs(harmless_inputs)
    #     # if is_sagemaker_mp_enabled():
    #     #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #     #     return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #         loss += self.compute_loss(model, harmless_inputs)

    #     del inputs  # Free memory

    #     kwargs = {}

    #     # Use correct learning rate for LOMO optimizers
    #     # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #     #     kwargs["learning_rate"] = self._get_learning_rate()

    #     # Handle multi-GPU training
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # Average loss across GPUs

    #     # Backward pass
    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss, **kwargs)

    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #                 param.grad.zero_()  #  Set gradient to zero

    #     return loss.detach() / self.args.gradient_accumulation_steps  # Continue normal training


class SecurityVectorTrainer(Trainer):
    # def init(self, harmless_dataset):
    #     self.harmless_dataloader = self.get_harmless_dataloader(harmless_dataset)
    #     self.harmless_data_iter = iter(self.harmless_dataloader)
    
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
        # idx = 0
        # for i, (name, param) in enumerate(model.named_parameters()):
        #     if 'lora_A' in name:
        #         # layer_idx = int(re.findall(r'\d+', name)[0])
        #         # if layer_idx < 7 or layer_idx > 12:
        #         #     continue
        #         lora_a = param
        #         lora_b_name = name.replace('lora_A', 'lora_B')
        #         lora_b = dict(model.named_parameters())[lora_b_name]
        #         delta_w = torch.mm(lora_b, lora_a)
        #         # model_weight_name = name.replace('.lora_A.default', '')
        #         # model_weight = dict(model.named_parameters())[model_weight_name].clone().detach().to(delta_w.device)
        #         # regularized_term = F.cosine_similarity(model_weight.view(-1), delta_w.view(-1), dim=-1)
        #         buffer_lora_a = self.bufferlora_dict[name.replace('.default', '')].clone().detach().to(delta_w.device)
        #         buffer_lora_b = self.bufferlora_dict[lora_b_name.replace('.default', '')].clone().detach().to(delta_w.device)
        #         buffer_delta_w = torch.mm(buffer_lora_b, buffer_lora_a)
        #         regularized_term = F.cosine_similarity(buffer_delta_w.view(-1), delta_w.view(-1), dim=-1)
        #         reg_loss += regul_lambda * torch.norm(regularized_term, p='fro').to(reg_loss.device) ** 2
        #         idx += 1
        # if self.model.training:
        #     wandb.log({"loss/reg_loss": reg_loss.item() / regul_lambda})
        loss = loss + reg_loss

        return (loss, outputs) if return_outputs else loss

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

class SignRegulTrainer(Trainer):
    def init(self, d_align, regul_lambda=1.0):
        self.d_align = d_align
        self.regul_lambda = regul_lambda

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculationg might be slightly inacurate when performing gradient accumulation.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
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
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
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

        if self.d_align is not None:
            reg_loss = torch.zeros(1).to(loss.device)
            lambda_reg = self.regul_lambda
            
            count = 0
            for name, module in self.model.named_modules():
                # LoRA 모듈 찾기: peft의 로라 모듈은 보통 lora_A/lora_B, scaling을 가짐
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    continue
                if "default" not in module.lora_A or "default" not in module.lora_B:
                    continue
                
                A = module.lora_A["default"].weight    # (r, in)
                B = module.lora_B["default"].weight    # (out, r)
                scaling = getattr(module, "scaling", 1.0)["default"]
                DW = (B @ A) * float(scaling)
                d_align = self.d_align[name[17:]+".weight"].to(DW.device, DW.dtype)  # Adjust key slicing as needed
                # reg_loss += lambda_reg * F.relu(-DW/d_align.to(DW.device)).mean().to(reg_loss.device)
                M = torch.sign(DW) != torch.sign(d_align)
                reg_loss += (M * (DW ** 2)).sum().to(reg_loss.device) / (M.sum().to(reg_loss.device) + 1e-7)
                count += 1
            
            loss = loss + lambda_reg * reg_loss/count

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


class AsFTTrainer(Trainer):
    # def init(self, harmless_dataset):
    #     self.harmless_dataloader = self.get_harmless_dataloader(harmless_dataset)
    #     self.harmless_data_iter = iter(self.harmless_dataloader)
    
    def init(self, regul_lambda, project_matrix):
        self.regul_lambda = regul_lambda
        self.project_matrix = project_matrix

    # def get_harmless_dataloader(self,harmless_dataset) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].

    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    #     training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
     
    #     from transformers.trainer_utils import seed_worker
    #     from transformers.trainer_pt_utils import LengthGroupedSampler
    #     from torch.utils.data import DataLoader, RandomSampler

    #     data_collator = self.data_collator
  
    #     sampler = RandomSampler(harmless_dataset)

    #     dataloader_params = {
    #         "batch_size": self.args.per_device_train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #     }

    #     if not isinstance(harmless_dataset, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = sampler
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker

    #     return self.accelerator.prepare(DataLoader(harmless_dataset, **dataloader_params))

    # def sample_from_harmless(self):
    #     # Get a  batch
    #     try:
    #         batch = next(self.harmless_data_iter)
    #     except (StopIteration):
    #         # If the iterator is exhausted, create a new iterator
    #         self.harmless_data_iter = iter(self.harmless_dataloader)
    #         batch = next(self.harmless_data_iter)
    #     return batch

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

        # reg_loss = torch.zeros_like(loss)
        # regul_lambda = self.regul_lambda
        # idx = 0
        # for i, (name, param) in enumerate(model.named_parameters()):
        #     if 'lora_A' in name:
        #         lora_a = param
        #         lora_b_name = name.replace('lora_A', 'lora_B')
        #         lora_b = dict(model.named_parameters())[lora_b_name]
        #         delta_w = torch.mm(lora_b, lora_a)
        #         # model_weight_name = name.replace('.lora_A.default', '')
        #         # model_weight = dict(model.named_parameters())[model_weight_name].clone().detach().to(delta_w.device)
        #         # regularized_term = F.cosine_similarity(model_weight.view(-1), delta_w.view(-1), dim=-1)
        #         buffer_lora_a = self.asft_dict[name.replace('.default', '')].clone().detach().to(delta_w.device)
        #         buffer_lora_b = self.asft_dict[lora_b_name.replace('.default', '')].clone().detach().to(delta_w.device)
        #         buffer_delta_w = torch.mm(buffer_lora_b, buffer_lora_a)
        #         regularized_term = F.cosine_similarity(buffer_delta_w.view(-1), delta_w.view(-1), dim=-1)
        #         reg_loss += regul_lambda * torch.norm(regularized_term, p='fro').to(reg_loss.device) ** 2
        #         idx += 1

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



# ========== SALoRA Components ==========

class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear for activation recording
    """
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = []  # offload to CPU
        self.record_activation = True

    def clear_act_buffer(self):
        self.activation_norms = []

    def forward(self, x):
        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask.to(x.device)]  # num * dim
            else:
                x_ = x  # bs * seq_len * dim
            self.activation_norms.append(
                x_.view(-1, x_.shape[-1]).cpu()
            )  # offload to CPU.
        out = self.base(x)
        return out


class set_mask:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = self.mask

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = None


def make_Act(model, verbose=False):
    replace_map = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


class AutoDeviceLinear(nn.Linear):
    """
    기존 nn.Linear를 상속받아, forward 시 input을 자동으로 
    weight가 위치한 디바이스로 이동시키는 Custom Linear 클래스입니다.
    """
    def forward(self, input):
        # 요청하신 핵심 로직: input을 weight의 device로 이동
        return F.linear(input.to(self.weight.device).to(dtype=self.weight.dtype), self.weight, self.bias)

def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    from functools import reduce
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            linear_module = module.base
            new_linear_module = AutoDeviceLinear(
                in_features=linear_module.in_features,
                out_features=linear_module.out_features,
                bias=(linear_module.bias is not None)
            )

            with torch.no_grad():
                new_linear_module.weight.copy_(linear_module.weight)
                if linear_module.bias is not None:
                    new_linear_module.bias.copy_(linear_module.bias)

            # [수정 3] 디바이스 이동 (기존 레이어가 있던 GPU로)
            new_linear_module.to(linear_module.weight.device)

            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            setattr(parent_module, name.split(".")[-1], new_linear_module)
    return model


def clear_act_buffer(act_model):
    for name, module in act_model.named_modules():
        if isinstance(module, ActLinear):
            module.clear_act_buffer()

from peft.tuners.lora import Linear as LoraLinear  # 원본 클래스

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

class SafePeftLinear(LoraLinear):
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        
        # [Step 1] Base Layer 계산 준비
        # 입력 x를 Base Layer(self.weight) 위치로 안전하게 이동
        target_device = self.weight.device
        if x.device != target_device:
            x_base = x.to(target_device)
        else:
            x_base = x

        # 어댑터가 없으면 Base만 계산하고 종료
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x_base, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x_base, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        elif self.r[self.active_adapter] > 0 and not self.merged:
            # 1. Base Layer 계산
            result = F.linear(x_base, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            # ---------------------------------------------------------------
            # [핵심 수정] LoRA A -> LoRA B 과정을 단계별로 쪼개서 이송 작전 수행
            # ---------------------------------------------------------------
            adapter = self.active_adapter
            layer_A = self.lora_A[adapter]
            layer_B = self.lora_B[adapter]
            dropout = self.lora_dropout[adapter]
            scaling = self.scaling[adapter]

            # 2. Input -> LoRA A (A 위치로 이동)
            # x를 layer_A가 있는 곳으로 납치
            x_a = x.to(device=layer_A.weight.device, dtype=layer_A.weight.dtype)
            out_a = layer_A(dropout(x_a))

            # 3. LoRA A Output -> LoRA B (B 위치로 이동)
            # A의 결과물(out_a)이 B의 위치와 다를 수 있으므로, B 위치로 강제 이동
            x_b = out_a.to(device=layer_B.weight.device, dtype=layer_B.weight.dtype)
            out_b = layer_B(x_b)

            # 4. Scaling 적용 (스칼라이므로 위치 상관없으나 안전하게 B 위치에서 계산)
            out_b = out_b * scaling

            # 5. Result 합산 (Base 결과 위치로 이동)
            # 최종 결과인 out_b를 result(Base Layer)가 있는 곳으로 가져와서 더함
            if out_b.device != result.device:
                out_b = out_b.to(result.device)
            
            result += out_b
            # ---------------------------------------------------------------

        else:
            result = F.linear(x_base, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result

class SALoRATrainer(Trainer):
    """
    Trainer that initializes LoRA weights using SALoRA (PiSSA) method from lora_train_act.py.
    This performs activation-based LoRA initialization before training.
    """
    
    def get_harmful_dataloader(self, harmful_dataset) -> DataLoader:
        """
        Returns the dataloader for harmful dataset (same as BoosterAlignmentTrainer).
        """
        from transformers.trainer_utils import seed_worker
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))
    
    def init(
        self, 
        harmful_dataset,
        tokenizer,
        peft_config,
        safe_dataset=None,
        rs=None,
        ds=None,
        n_iter=30,
        seqlen=4096,
        modelname=None
    ):
        """
        Initialize SALoRA trainer and perform LoRA initialization.
        Uses provided LoraConfig from train.py.
        
        Args:
            harmful_dataset: Harmful dataset (SupervisedDataset) - same as Booster uses
            tokenizer: Tokenizer for the model
            peft_config: LoraConfig from train.py
            safe_dataset: Safety dataset (SupervisedDataset). If None, uses harmful_dataset (for testing)
            rs: Rank for safety dataset SVD (default: same as LoRA rank)
            ds: Rank for training dataset SVD (default: same as LoRA rank)
            n_iter: Number of iterations for SVD
            seqlen: Sequence length
            modelname: Model name for data loading
        """
        self.clock = 0
        self.steps = 0
        self.statistic = 0
        
        self.tokenizer = tokenizer
        self.n_iter = n_iter
        self.seqlen = seqlen
        # Auto-detect modelname from model config (SALoRA는 항상 일반 모델을 받음)
        base_model = self.model
        if modelname is None:
            # Try to detect from model config
            model_type = base_model.config.model_type.lower()
            name_or_path = str(getattr(base_model.config, 'name_or_path', '')).lower()
            if 'llama3' in model_type or 'llama3' in name_or_path:
                self.modelname = "llama3"
            elif 'qwen' in model_type or 'qwen' in name_or_path:
                self.modelname = "qwen"
            elif 'gemma' in model_type or 'gemma' in name_or_path:
                self.modelname = "gemma"
            else:
                # Default to llama2 format
                self.modelname = "llama3"
                print(f"Warning: Could not detect model type from {model_type}, defaulting to llama3 format")
        else:
            self.modelname = modelname
        self.harmful_dataset = harmful_dataset
        # If safe_dataset is None, use harmful_dataset (for testing)
        self.safe_dataset = safe_dataset if safe_dataset is not None else harmful_dataset
        
        # train.py에서 전달받은 LoraConfig 사용
        self.peft_config = peft_config
        self.rank = peft_config.r
        self.target_module_list = peft_config.target_modules
        
        # target_modules에 따라 more_module 결정
        if set(self.target_module_list) == {"q_proj", "v_proj"}:
            self.more_module = False
        elif set(self.target_module_list) == {"q_proj", "gate_proj", "v_proj", "up_proj", "down_proj"}:
            self.more_module = True
        else:
            self.more_module = False
        
        # rs, ds는 기본적으로 rank와 동일
        self.rs = rs if rs is not None else self.rank
        self.ds = ds if ds is not None else self.rank
        
        self.divide_num = len(self.target_module_list)
        
        # Perform LoRA initialization
        self._initialize_lora()
        
    def _initialize_lora(self):
        """
        Perform actual LoRA initialization using SALoRA method.
        Uses harmful_dataset (same as Booster) for training dataset.
        원본 코드 순서: 일반 모델 → make_Act → activation 수집 → base weight 수정 → revert_Act_to_Linear → get_peft_model
        """
        # 원본 코드 순서: 일반 모델 → make_Act → activation 수집 → base weight 수정 → revert_Act_to_Linear → get_peft_model
        print("Starting SALoRA LoRA initialization with base model...")
        model = self.model  # 일반 모델 사용
        t1 = time.time()
        
        # Convert model to ActLinear for activation recording
        model = make_Act(model, verbose=False)
        model.requires_grad_(False)
        if not hasattr(model, 'seqlen'):
            model.seqlen = self.seqlen
        clear_act_buffer(model)
        
        # Disable recording initially
        for name, module in model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False
                module.clear_act_buffer()
        
        # Prepare harmful dataset dataloader (same as Booster uses)
        print("Preparing harmful dataset dataloader...")
        harmful_dataloader = self.get_harmful_dataloader(self.harmful_dataset)
        
        # Prepare safety dataset dataloader (use safe_dataset if provided, otherwise harmful_dataset)
        print("Preparing safety dataset dataloader...")
        safe_dataloader = self.get_harmful_dataloader(self.safe_dataset)
        
        num_hidden_layers = model.config.num_hidden_layers
        weight_list = {}
        current_num = -1
        
        # First pass: collect activations from safety dataset
        print("Collecting activations from safety dataset...")
        for layer in range(num_hidden_layers):
            layer_filter_fn = lambda x: f"layers.{layer}." in x
            
            # Enable recording for current layer
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    module.record_activation = True
                    module.clear_act_buffer()
            
            # Forward pass on safety dataset (SupervisedDataset format: dict with input_ids, labels, attention_mask)
            for batch in safe_dataloader:
                inp = batch['input_ids'].to(model.device)
                tar = batch['labels'].to(model.device)
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)
            
            # Process activations and compute safety V
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    f_name = ""
                    for t_name in self.target_module_list:
                        if t_name in name:
                            f_name = t_name
                            break
                    if len(f_name) == 0:
                        continue
                    
                    current_num += 1
                    layer_n = f_name
                    module.activation_norms = torch.cat(module.activation_norms, dim=0).to(model.device)
                    score = module.activation_norms @ module.base.weight.data.T.to(model.device)
                    d_out, d_in = module.base.weight.data.shape
                    total_rank = min(d_out, d_in)
                    
                    print(f"Safety SVD for {name}: rank {total_rank - self.rank} / {total_rank}")
                    
                    # SVD for safety dataset
                    U, S, V = torch.svd_lowrank(score.float(), q=self.rs, niter=self.n_iter)
                    V = V.type(module.base.weight.data.dtype)
                    weight_list[layer_n + '_' + str(current_num // self.divide_num) + '_V'] = V
            
            # Disable recording for current layer
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    module.record_activation = False
                    module.clear_act_buffer()
        
        # Second pass: collect activations from training dataset and initialize LoRA
        print("Collecting activations from training dataset and initializing LoRA...")
        current_num = -1
        for layer in range(num_hidden_layers):
            layer_filter_fn = lambda x: f"layers.{layer}." in x
            
            # Enable recording for current layer
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    module.record_activation = True
            
            # Forward pass on training dataset (harmful_dataset)
            for batch in harmful_dataloader:
                # SupervisedDataset format: dict with input_ids, labels, attention_mask
                inp = batch['input_ids'].to(model.device)
                tar = batch['labels'].to(model.device)
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)
            
            # Process activations and initialize LoRA weights
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    f_name = ""
                    for t_name in self.target_module_list:
                        if t_name in name:
                            f_name = t_name
                            break
                    if len(f_name) == 0:
                        continue
                    
                    current_num += 1
                    layer_n = f_name
                    module.activation_norms = torch.cat(module.activation_norms, dim=0).to(model.device)
                    score = module.activation_norms @ module.base.weight.data.T.to(model.device)
                    d_out, d_in = module.base.weight.data.shape
                    total_rank = min(d_out, d_in)
                    
                    print(f"Training SVD for {name}: rank {total_rank - self.rank} / {total_rank}")
                    
                    # SVD for training dataset
                    U, S, V = torch.svd_lowrank(score.float(), q=self.ds, niter=self.n_iter)
                    V = V.type(module.base.weight.data.dtype)
                    
                    # SVD for base weight
                    U2, S2, V2 = torch.svd_lowrank(module.base.weight.data.float(), q=self.rank, niter=self.n_iter)
                    U2 = U2.type(module.base.weight.data.dtype)
                    S2 = S2.type(module.base.weight.data.dtype)
                    V2 = V2.type(module.base.weight.data.dtype)
                    
                    # Get safety V
                    safeV = weight_list[layer_n + '_' + str(current_num // self.divide_num) + '_V']
                    
                    # Compute LoRA components
                    weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_C'] = (
                        torch.eye(V.size(0), dtype=module.base.weight.dtype, device=module.base.weight.device) 
                        - (safeV @ safeV.T).to(module.base.weight.device)
                    )
                    
                    weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_B'] = U2 @ torch.diag(torch.sqrt(S2))
                    weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_B'] = (
                        V @ V.T @ weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_B'].to(V.device)
                    )
                    weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_A'] = (
                        torch.diag(torch.sqrt(S2)) @ V2.T
                    )
                    
                    # Update base weight
                    module.base.weight.data.sub_(
                        weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_C'].type(module.base.weight.data.dtype).to(module.base.weight.device)
                        @ weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_B'].type(module.base.weight.data.dtype).to(module.base.weight.device) 
                        @ weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'lora_A'].type(module.base.weight.data.dtype).to(module.base.weight.device)
                    )
                    weight_list[layer_n + '_' + str(current_num // self.divide_num) + 'weight'] = module.base.weight.data
            
            # Disable recording for current layer
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    module.record_activation = False
                    module.clear_act_buffer()
        
        t2 = time.time()
        print(f"SALoRA initialization time: {t2 - t1:.2f} seconds")
        
        # Revert ActLinear back to Linear
        model = revert_Act_to_Linear(model)
        model.zero_grad()
        
        # Store weight_list for later use in setting LoRA weights
        self.weight_list = weight_list
        
        # 원본 코드 순서: revert_Act_to_Linear() 후에 get_peft_model() 호출
        # 수정된 base weight를 가진 일반 모델로 PEFT 모델 생성
        print("Converting model to PEFT model (after base weight modification)...")
        from peft import get_peft_model
        import peft.tuners.lora as peft_lora
        peft_lora.Linear = SafePeftLinear
        model = get_peft_model(model, self.peft_config)
        model.to(torch.bfloat16)
        model.print_trainable_parameters()
        
        # self.model을 PEFT 모델로 업데이트 (Trainer가 사용할 수 있도록)
        self.model = model
        
        print("SALoRA LoRA initialization completed.")
        
        # Set LoRA weights from SALoRA initialization
        self.set_lora_weights_from_salora()
        
    def sample_from_harmful(self):
        """Sample from harmful dataset (same as BoosterAlignmentTrainer)."""
        if not hasattr(self, 'harmful_data_iter'):
            self.harmful_dataloader = self.get_harmful_dataloader(self.harmful_dataset)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch
        
    def set_lora_weights_from_salora(self):
        """
        Set LoRA weights from SALoRA initialization. 
        This should be called after get_peft_model() to set the initialized weights.
        """
        if not hasattr(self, 'weight_list') or not self.weight_list:
            raise ValueError("SALoRA weights not initialized. Call init() first.")
        
        current_num = 0
        print("Setting LoRA weights from SALoRA initialization...")
        
        for name, module in self.model.named_modules():
            for t_name in self.target_module_list:
                if t_name in name:
                    if 'lora_A.default' in name:
                        key = t_name + '_' + str(current_num // (2 * self.divide_num)) + 'lora_A'
                        if key in self.weight_list:
                            module.weight.data = self.weight_list[key]
                            current_num += 1
                    if 'lora_B.default' in name:
                        key = t_name + '_' + str(current_num // (2 * self.divide_num)) + 'lora_B'
                        if key in self.weight_list:
                            module.weight.data = self.weight_list[key]
                            current_num += 1
                    if t_name + '.' not in name and hasattr(module, 'lora_C'):
                        key = t_name + '_' + str(current_num // 4) + 'lora_C'
                        if key in self.weight_list:
                            module.lora_C = self.weight_list[key].clone()
                            module.requires_grad = False
                    break
        self.weight_list['divide_num'] = self.divide_num
        torch.save(self.weight_list, self.args.output_dir+'/lora_ABC.pt')
        del self.weight_list
        
        # 확인: LoRA 파라미터의 requires_grad 상태 확인
        print("\n=== LoRA 파라미터 requires_grad 상태 확인 ===")
        trainable_count = 0
        non_trainable_count = 0
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                if param.requires_grad:
                    trainable_count += 1
                    if trainable_count <= 5:  # 처음 5개만 출력
                        print(f"  [TRAINABLE] {name}: requires_grad={param.requires_grad}, shape={param.shape}")
                else:
                    non_trainable_count += 1
                    if non_trainable_count <= 5:  # 처음 5개만 출력
                        print(f"  [NON-TRAINABLE] {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        print(f"  총 LoRA trainable 파라미터: {trainable_count}")
        print(f"  총 LoRA non-trainable 파라미터: {non_trainable_count}")
        
        # 확인: 전체 trainable 파라미터 개수
        total_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        total_params = sum(1 for p in self.model.parameters())
        print(f"  전체 trainable 파라미터: {total_trainable}/{total_params}")
        print("=" * 50 + "\n")
        
        print("LoRA weights set from SALoRA initialization.")
    
    # def training_step(
    #     self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    # ) -> torch.Tensor:
    #     """
    #     Perform a training step. Uses standard Trainer training step.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     def step(step_inputs):
    #         if is_sagemaker_mp_enabled():
    #             loss_mb = smp_forward_backward(model, step_inputs, self.args.gradient_accumulation_steps)
    #             return loss_mb.reduce_mean().detach().to(self.args.device)

    #         with self.compute_loss_context_manager():
    #             # model_inputs = {k: v.to(model.device) for k, v in step_inputs.items()}
    #             model_inputs = {}
    #             model_inputs["input_ids"] = step_inputs["input_ids"].to(model.device)
    #             model_inputs["attention_mask"] = step_inputs["attention_mask"].to(model.device)
    #             model_inputs["labels"] = step_inputs["labels"].to(model.device)
    #             loss = self.compute_loss(model, model_inputs)
            
    #         # 확인: loss의 requires_grad 상태 확인 (첫 번째 step에서만)
    #         if not hasattr(self, '_first_step_checked'):
    #             print("\n=== Loss 상태 확인 ===")
    #             print(f"  loss type: {type(loss)}")
    #             print(f"  loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
    #             print(f"  loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'N/A'}")
    #             print(f"  loss grad_fn: {loss.grad_fn if hasattr(loss, 'grad_fn') else 'N/A'}")
                
    #             # 확인: 모델의 trainable 파라미터 확인
    #             trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    #             print(f"  모델 trainable 파라미터 개수: {len(trainable_params)}")
    #             if len(trainable_params) == 0:
    #                 print("  [경고] trainable 파라미터가 없습니다!")
    #             else:
    #                 print(f"  처음 5개 trainable 파라미터: {trainable_params[:5]}")
    #             print("=" * 50 + "\n")
    #             self._first_step_checked = True
            
    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         if self.use_apex:
    #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             self.accelerator.backward(loss)
    #         return loss 
        
    #     loss = step(inputs)    
    #     return loss.detach() / self.args.gradient_accumulation_steps
