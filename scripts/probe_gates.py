"""The size of the adaLN gates of a checkpoint, before any bound: per block, `mod + table` in the gate rows at the
score's 10 t, read as the RMS over channels and t, the largest |gate|, and the share of channels with |gate| > 1, where
a tanh would start to bend it; then the RMS of each part, the shared projection's and each block's table's.
CPU, fp32: it runs beside a training run.

usage: probe_gates.py RUN_DIR STEP
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.model import GATE_ROWS, N_MOD, timestep_sinusoid
from nanodit.objective import t_quantiles
from train import SCORE_SHIFT, SCORE_TS, Config, build_model


@torch.no_grad()
def main(run_dir: str, step: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    model = build_model(cfg, torch.device("cpu"), torch.float32)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    t = t_quantiles(SCORE_TS, SCORE_SHIFT)
    mod = model.ada(model.time_mlp(timestep_sinusoid(t))).view(len(t), N_MOD, model.width)
    shared = mod[:, list(GATE_ROWS)]
    print("shared projection's gates, RMS: " + " / ".join(f"{g.square().mean().sqrt():.3f}" for g in shared.unbind(1)))
    print(f"{run.name} step {step}: per block, mixer-branch gate then MLP-branch gate, as RMS / max |g| / share |g| > 1")
    for i, block in enumerate(model.blocks):
        gates = (mod + block.table)[:, list(GATE_ROWS)]  # [t, 2, width]
        reads = [f"{g.square().mean().sqrt():.3f} / {g.abs().max():.2f} / {(g.abs() > 1).float().mean():.4f}" for g in gates.unbind(1)]
        table = " / ".join(f"{row.square().mean().sqrt():.3f}" for row in block.table[list(GATE_ROWS)])
        print(f"{i:2d} {"conv" if block.is_conv else "attn"} " + "   ".join(reads) + f"   table RMS {table}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
