"""Where each conv block's depthwise kernel puts its weight: the share of the kernel's squared weight on each ring
of taps (ring r: the taps at Chebyshev distance r from the centre, r = 0 the centre), summed over channels, against
the share a kernel drawn at init would have (every tap the same variance: the ring's tap count over k²). A ring
above its init share is reach the block uses. Run with the model code the checkpoint was trained on.
CPU, fp32: it runs beside a training run.

usage: probe_conv_taps.py RUN_DIR STEP
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from train import Config, build_model


@torch.no_grad()
def main(run_dir: str, step: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    model = build_model(cfg, torch.device("cpu"), torch.float32)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    print(f"{run.name} step {step}: per conv block, kernel, then per ring the share of squared weight (init share)")
    for i, block in enumerate(model.blocks):
        if not block.is_conv:
            continue
        w = block.conv.weight[:, 0]  # [width, k, k]
        k = w.shape[-1]
        pos = torch.arange(k) - k // 2
        ring = torch.maximum(pos.abs()[:, None], pos.abs()[None, :])  # [k, k]
        energy = w.square().sum(0)
        shares = [f"{energy[ring == r].sum() / energy.sum():.3f} ({(ring == r).sum().item() / k**2:.3f})" for r in range(k // 2 + 1)]
        print(f"{i:2d} {k}x{k}  " + "  ".join(shares), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
