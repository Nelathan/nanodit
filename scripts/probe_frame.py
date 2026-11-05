"""Does Usui's frame on one matrix rotate into new directions or stay in its subspace? Reads a run's
checkpoints; prints, writes nothing.

usage: probe_frame.py RUN_DIR PARAM   (a matrix Usui tracks, e.g. blocks.0.qkv.weight)

Per checkpoint: W's stable rank Σs²/s₁² and its singular values above 10% of s₁; the share of ‖W‖² outside
the current frame (0: W lives in today's frame); ‖F_prevᵀ F‖²/r against the previous checkpoint's frame
(1: the same subspace). Run it between checkpoints of a live run: one being written may be read half-done.
"""
import glob
import sys
import tomllib

import torch
from safetensors.torch import load_file
from usuitrack.frame import resolve_side

from tinydit.model import DiT
from tinydit.optim import build_optimizers

run, name = sys.argv[1], sys.argv[2]
with open(f"{run}/config.toml", "rb") as f:
    depth = tomllib.load(f)["depth"]
model = DiT(torch.zeros(768), depth)
matrix, _ = build_optimizers(model, 1e-3, 0.95, 0.1, 1e-3, pin_weights=False)
# optimizer.pt keys its state by position across the param groups
placed = [(id(p), g["side"]) for g in matrix.param_groups for p in g["params"]]
index = [key for key, _ in placed].index(id(model.get_parameter(name)))
side = resolve_side(model.get_parameter(name).shape, placed[index][1])
previous = None
print(f"{name}: step  stable_rank  above_10%  outside_frame  overlap_prev")
for checkpoint in sorted(glob.glob(f"{run}/checkpoints/step_*")):
    W = load_file(f"{checkpoint}/model.safetensors")[name].float()
    F = torch.load(f"{checkpoint}/optimizer.pt", map_location="cpu")[0]["state"][index]["frame"].float()
    in_frame = W @ F if side == "right" else F.T @ W  # the frame spans W's input (right) or output (left) side
    s = torch.linalg.svdvals(W)
    outside = 1 - in_frame.norm() ** 2 / W.norm() ** 2
    overlap = (previous.T @ F).norm() ** 2 / F.shape[1] if previous is not None else float("nan")
    print(f"{int(checkpoint[-7:]):>6} {(s**2).sum() / s[0]**2:12.1f} {(s > 0.1 * s[0]).sum():10d} {outside:14.3f} {overlap:13.3f}")
    previous = F
