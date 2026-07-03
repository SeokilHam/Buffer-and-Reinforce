<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<h1 align="center">Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models</h1>

**Authors:** Seokil Ham, Jaehyuk Jang, Wonjun Lee, Changick Kim

**Paper:** [https://arxiv.org/abs/2411.15224](https://arxiv.org/abs/2605.24550)

> **[Abstract]** Fine-tuning-as-a-Service enables personalization of large language models (LLMs), but it can weaken safety-alignment under harmful fine-tuning attacks. Recent work has shown that activating harmful-behavior modules during fine-tuning can prevent models from learning undesired behaviors, but its mechanism remains unclear. In this paper, we revisit temporary jailbreaking as a defense against harmful fine-tuning and provide a gradient-level analysis showing that it saturates safety-degrading gradients while preserving benign task-relevant gradients. Based on this insight, we propose a **Buffer-and-Reinforce fine-tuning framework** that buffers harmful updates during user fine-tuning and reinforces safety after adaptation. Specifically, BufferLoRA induces temporary jailbreaking as a removable adapter to reduce harmful updates during user fine-tuning. After adaptation, ReinforceLoRA, trained to recover refusal behavior under the temporarily jailbroken state, is integrated with UserLoRA via QR decomposition-based merging to reinforce safety while preserving user-task performance. Extensive experiments show that our framework achieves superior safety and utility with no additional safety data during user fine-tuning and minimal computational cost.


<div align="center">
  <img src="" width="80%"/>
</div>




## Installation

Requirements:
- Linux
- NVIDIA GPU
- PyTorch 2.1.0+
- CUDA 11.8+

```sh
conda create -n Buffer_and_Reinforce python=3.9
pip install -r requirements.txt
```

## Huggingface access
You should be able to access the model, but you first need to enter your token in the file `huggingface_token.txt`.


## Overall Framework
### Before User Fine-tuning
[Pretrained BufferLoRA and ReinforceLoRA] https://huggingface.co/SeokilH/Buffer-and-Reinforce
#### 1. Train BufferLoRA 
```
cd Buffer-and-Reinforce/script/finetune
bash bufferlora_llama.sh
```
#### 2. Train ReinforceLoRA
```
cd Buffer-and-Reinforce/script/finetune
bash reinforcelora_llama.sh
```
### User Fine-tuning & QR-Merging
```
cd Buffer-and-Reinforce/script/finetune
bash bufferlora_ft_llama.sh
```

## Citation

```
@article{ham2026jailbreak,
  title={Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models},
  author={Ham, Seokil and Jang, Jaehyuk and Lee, Wonjun and Kim, Changick},
  journal={arXiv preprint arXiv:2605.24550},
  year={2026}
}
```

## Reference
This codebase was partially adapted from the following repository:

- (https://github.com/git-disl/Booster)

We thank the authors for open-sourcing their work.






