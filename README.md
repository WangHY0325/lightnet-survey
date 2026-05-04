# LightNet-Survey: Reproduction Code

Code for controlled reproductions in the survey paper "A Survey of Lightweight Neural Networks."

## Directory Map

| Directory | Paper Section | Content |
|-----------|--------------|---------|
| `nas_imagewoof/` | Section II (NAS) | MobileNet family fine-tuning and A30 latency benchmark |
| `quant_llm/` | Section V (Quantization) | PTQ W8A8 and QAT W8A8 (LSQ+) evaluation on 30M LLM |
| `lowrank_vision/` | Section VI (Low-Rank) | SVD decomposition + BN recalibration + recovery training on MobileNetV2/ImageWoof |
| `peft_llm/` | Section VI (PEFT) | LoRA, DoRA, MoRA, and AdaLoRA training on LLaMA-2-7B-chat/Alpaca |

## Environment

All experiments were run with PyTorch 2.5.1 (CUDA 12.4) on NVIDIA A30 GPUs, using the conda environment `LightNet`.

SLURM scripts and full training logs are available upon request.
