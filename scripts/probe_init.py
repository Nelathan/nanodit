"""adaLN shift/scale magnitudes at init (PLAN F2), for the null embedding and random unit texts; prints."""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from tinydit.model import GATE_ROWS, N_MOD, TEXT_DIM, DiT, timestep_sinusoid

torch.manual_seed(0)
null = load_file("/mnt/mars/ai/tiny-dit/precomputed/null_embedding.safetensors")["null"]
model = DiT(null, depth=12, width=512)
t = torch.linspace(0.02, 0.98, 16)
shift_scale = [r for r in range(N_MOD) if r not in GATE_ROWS]
with torch.no_grad():
    for name, text in [("null embedding", null.expand(16, -1)), ("random unit", torch.nn.functional.normalize(torch.randn(16, TEXT_DIM), dim=-1))]:
        c = model.time_mlp(timestep_sinusoid(t)) + model.text_proj(F.rms_norm(text, (TEXT_DIM,)))
        mod = model.ada(c).view(-1, N_MOD, model.width)
        block = mod[:, shift_scale] + model.blocks[0].table[shift_scale]
        print(f"{name}: c std {c.std():.3f}; block 0 shift/scale std {block.std():.3f}")
