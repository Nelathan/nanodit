"""Euler sampling with classifier-free guidance. The network gives F; v̂ = x_t - F."""

import torch
from torch import Tensor

from .model import Prompt


@torch.no_grad()
def euler_sample(model, noise: Tensor, prompt: Prompt, steps: int, guidance: float) -> Tensor:
    """Integrate from t=1 (pure noise) to t=0 in `steps` equal steps. CFG is applied to F,
    which is the same as applying it to v̂ because v̂ = x_t - F is affine in F. At guidance 1 it is the conditional
    prediction alone, and the null branch is not run."""
    x = noise
    ts = torch.linspace(1, 0, steps + 1, device=noise.device)
    conditional = guidance == 1.0
    if not conditional:
        null = model.null_prompt(*prompt.tokens.shape[:2])
        both = Prompt(*(torch.cat([a, b]) for a, b in zip(prompt, null)))
    for t, t_next in zip(ts[:-1], ts[1:]):
        if conditional:
            out = model(x, t.expand(x.shape[0]), prompt)
        else:
            out_text, out_null = model(torch.cat([x, x]), t.expand(2 * x.shape[0]), both).chunk(2)
            out = out_null + guidance * (out_text - out_null)
        x = x + (t_next - t) * (x - out)
    return x
