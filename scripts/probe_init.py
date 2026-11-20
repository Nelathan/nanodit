"""adaLN shift/scale magnitudes at init (PLAN F2) over a t grid; prints.

usage: probe_init.py NULL_EMBEDDING.safetensors
"""
import sys

import torch
from safetensors.torch import load_file

from nanodit.model import GATE_ROWS, N_MOD, DiT, timestep_sinusoid

torch.manual_seed(0)
model = DiT(load_file(sys.argv[1])["null_tokens"], depth=12, width=512)
t = torch.linspace(0.02, 0.98, 16)
shift_scale = [r for r in range(N_MOD) if r not in GATE_ROWS]
with torch.no_grad():
    c = model.time_mlp(timestep_sinusoid(t))
    mod = model.ada(c).view(-1, N_MOD, model.width)
    block = mod[:, shift_scale] + model.blocks[0].table[shift_scale]
    print(f"c std {c.std():.3f}; block 0 shift/scale std {block.std():.3f}")
