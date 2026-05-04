"""
W8A8 PTQ Evaluation Script for 30M LLM
Post-Training Quantization using simple uniform quantization
"""

import torch
import torch.nn as nn
import sys
import os
import json
from argparse import Namespace
from pathlib import Path

# Add QuEST src to path
quest_src = "/gpool/home/wanghongyang/WangHY/QuEST/src"
sys.path.insert(0, quest_src)

from data.utils import DataReader
from models.utils import get_model


class UniformPTQQuantizer(nn.Module):
    """Simple uniform PTQ quantizer for W8A8"""
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.n_levels = 2 ** bits
        self.scale = None

    def calibrate(self, x):
        """Calibrate scale based on max absolute value"""
        max_val = torch.max(torch.abs(x))
        self.scale = max_val / (self.n_levels / 2 - 1)

    def forward(self, x):
        if self.scale is None:
            self.calibrate(x)
        # Quantize
        x_q = torch.round(x / self.scale).clamp(-(self.n_levels // 2), self.n_levels // 2 - 1)
        # Dequantize
        return x_q * self.scale


def quantize_model_ptq(model, calibration_data, device='cuda'):
    """Apply PTQ to model using calibration data"""
    print("Applying W8A8 PTQ quantization...")

    # Collect all linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    print(f"Found {len(linear_layers)} linear layers to quantize")

    # Calibrate weights (static)
    for name, layer in linear_layers:
        quantizer = UniformPTQQuantizer(bits=8)
        quantizer.calibrate(layer.weight.data)
        layer.weight.data = quantizer(layer.weight.data)

    # Calibrate activations (need forward pass)
    activation_scales = {}
    hooks = []

    def get_activation_hook(name):
        def hook(module, input, output):
            if name not in activation_scales:
                quantizer = UniformPTQQuantizer(bits=8)
                quantizer.calibrate(input[0])
                activation_scales[name] = quantizer
        return hook

    # Register hooks
    for name, layer in linear_layers:
        hook = layer.register_forward_hook(get_activation_hook(name))
        hooks.append(hook)

    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model(x, y)
            break  # Only use one batch for calibration

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Apply activation quantization via forward pre-hook
    # PyTorch pre-hook signature: hook(module, args) -> modified_args or None
    # where args is a tuple of positional inputs
    def quantize_activation_hook(name):
        def hook(module, args):
            if not isinstance(args, tuple):
                args = (args,)
            if name in activation_scales:
                return (activation_scales[name](args[0]),) + args[1:]
            return args
        return hook

    for name, layer in linear_layers:
        layer.register_forward_pre_hook(quantize_activation_hook(name))

    print("PTQ quantization completed")
    return model


@torch.no_grad()
def evaluate(model, data_reader, device='cuda', max_iters=100):
    """Evaluate model using DataReader.
    Llama.forward returns dict: {"logits": ..., "loss": ...}
    When called with targets and get_logits=True, both are available.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for i in range(max_iters):
        x, y = data_reader.sample_batch()
        x, y = x.to(device), y.to(device)

        out = model(x, y, get_logits=True)
        logits = out["logits"]
        total_loss += out["loss"].item()

        pred = logits.argmax(dim=-1)
        total_correct += (pred == y).sum().item()
        total_tokens += y.numel()

    avg_loss = total_loss / max_iters
    accuracy = total_correct / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }


def main():
    print("=== W8A8 PTQ Evaluation ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model configuration (30M) — must match the checkpoint's training config
    args = Namespace(
        model="llama",
        n_layer=6,
        n_embd=640,
        n_head=5,
        multiple_of=256,
        sequence_length=2048,
        vocab_size=50304,
        init_std=0.02,
        dropout=0.0,
        rmsnorm_eps=1e-5,
        bias=False,
        compile=False,
        parallel_block=False,
        mlp_dim_exp_factor=1.0,
        dtype="bfloat16",

        w_quant="NoQuantizer",
        a_quant="NoQuantizer",
        w_quant_kwargs={},
        a_quant_kwargs={},
        use_pretrained="none",
    )

    print(f"Model config: {args.n_layer} layers, {args.n_embd} embd, {args.n_head} heads")

    # Load FP32 checkpoint
    checkpoint_path = "/gpool/home/wanghongyang/WangHY/QuEST/data/fineweb_llama_nlayers6_nhead5_lr0.0012_sched_cos_warmup286_decay_linear_0.1_iter2861_bs8x64_ws1_seed42_data_seed1337/final_model.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model via QuEST factory
    model = get_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Reference model loaded successfully")

    # Load FineWeb dataset via DataReader (pre-tokenized .npy files)
    print("Loading FineWeb dataset...")
    fineweb_dir = "/gpool/home/wanghongyang/WangHY/QuEST/datasets/fineweb"
    batch_size = 8

    train_reader = DataReader(
        data_src=os.path.join(fineweb_dir, "train_tokens.npy"),
        batch_size=batch_size,
        sequence_length=2048,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
    )

    val_reader = DataReader(
        data_src=os.path.join(fineweb_dir, "val_tokens.npy"),
        batch_size=batch_size,
        sequence_length=2048,
        seed=1337,
        with_replacement=False,
        auto_shard=False,
    )

    # Evaluate FP32 baseline
    print("\n=== Reference Checkpoint ===")
    fp32_results = evaluate(model, val_reader, device, max_iters=100)
    print(f"Reference - Loss: {fp32_results['val_loss']:.4f}, "
          f"Perplexity: {fp32_results['val_perplexity']:.4f}, "
          f"Accuracy: {fp32_results['val_accuracy']:.6f}")

    # Apply PTQ (calibration uses one batch from train_reader)
    print("\n=== Applying PTQ ===")
    x_cal, y_cal = train_reader.sample_batch()
    model = quantize_model_ptq(model, [(x_cal, y_cal)], device)

    # Evaluate PTQ model
    print("\n=== W8A8 PTQ Results ===")
    ptq_results = evaluate(model, val_reader, device, max_iters=100)
    print(f"PTQ W8A8 - Loss: {ptq_results['val_loss']:.4f}, "
          f"Perplexity: {ptq_results['val_perplexity']:.4f}, "
          f"Accuracy: {ptq_results['val_accuracy']:.6f}")

    # Calculate degradation
    loss_diff = ptq_results['val_loss'] - fp32_results['val_loss']
    ppl_diff = ptq_results['val_perplexity'] - fp32_results['val_perplexity']

    print("\n=== Degradation ===")
    print(f"Loss increase: +{loss_diff:.4f}")
    print(f"Perplexity increase: +{ppl_diff:.4f}")

    # Save results
    results = {
        'fp32': fp32_results,
        'ptq_w8a8': ptq_results,
        'degradation': {
            'loss_diff': loss_diff,
            'ppl_diff': ppl_diff
        }
    }

    output_dir = Path("results/ptq_w8a8")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
