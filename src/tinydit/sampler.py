"""Euler sampling with classifier-free guidance. The network gives F; v̂ = x_t - F."""

import torch
from torch import Tensor


@torch.no_grad()
def euler_sample(model, noise: Tensor, text: Tensor, steps: int, guidance: float) -> Tensor:
    """Integrate from t=1 (pure noise) to t=0 in `steps` equal steps. CFG is applied to F,
    which is the same as applying it to v̂ because v̂ = x_t - F is affine in F."""
    x = noise
    null = model.null_embedding.expand_as(text)
    ts = torch.linspace(1, 0, steps + 1, device=noise.device)
    for t, t_next in zip(ts[:-1], ts[1:]):
        out_text, out_null = model(torch.cat([x, x]), t.expand(2 * x.shape[0]), torch.cat([text, null])).chunk(2)
        out = out_null + guidance * (out_text - out_null)
        x = x + (t_next - t) * (x - out)
    return x
