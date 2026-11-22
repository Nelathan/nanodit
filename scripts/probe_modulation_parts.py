"""What the adaLN modulation of a checkpoint is made of. Each row (the mixer's and the MLP's read scale and write gate)
is M[block, t, channel] = mod(t) + table(block), exactly a per-channel constant plus a block part plus a t part:
M = μ(c) + A(block, c) + B(t, c), A the tables about their mean over blocks, B the shared projection about its mean
over t. Per row and part (slice: the first TOKEN_DIM channels; meta: the rest), each part's share of the mean square
of M; then how much of the block part A one scalar per block and part carries, and how much its first singular pair
(one channel profile scaled per block) carries. t runs over 20 evenly spaced values. CPU, fp32.

usage: probe_modulation_parts.py RUN_DIR STEP
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.model import N_MOD, TOKEN_DIM, timestep_sinusoid
from nanodit.objective import t_uniform
from train import Config, build_model

ROW_NAMES = ("mixer read scale", "mixer gate", "MLP read scale", "MLP gate")


@torch.no_grad()
def main(run_dir: str, step: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    model = build_model(cfg, torch.device("cpu"), torch.float32)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    t = t_uniform(20)
    mod = model.ada(model.time_mlp(timestep_sinusoid(t))).view(len(t), N_MOD, model.width)
    tables = torch.stack([block.table for block in model.blocks])  # [block, N_MOD, width]
    print(f"{run.name} step {step}: per row and part, the share of mean M² in the constant μ / the block part A / the t "
          "part B; of A, the share one scalar per block carries / its first singular pair carries; RMS of M")
    for row, name in enumerate(ROW_NAMES):
        reads = []
        for part in (slice(0, TOKEN_DIM), slice(TOKEN_DIM, None)):
            table, shared = tables[:, row, part], mod[:, row, part]  # [block, c], [t, c]
            a, b = table - table.mean(0), shared - shared.mean(0)
            mu = table.mean(0) + shared.mean(0)
            total = mu.square().mean() + a.square().mean() + b.square().mean()  # the cross terms vanish
            scalar = a.mean(1, keepdim=True).square().mean() / a.square().mean()
            first = torch.linalg.svdvals(a)[0].square() / a.square().sum()
            reads.append(f"{mu.square().mean() / total:.2f} / {a.square().mean() / total:.2f} / {b.square().mean() / total:.2f}"
                         f"; {scalar:.2f} / {first:.2f}; {total.sqrt():.3f}")
        print(f"{name:16s} slice {reads[0]}   meta {reads[1]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
