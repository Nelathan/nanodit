"""SPEC §4: x0 velocity on rectified flow, x_t = (1 - t)·x + t·ε."""

import math

import torch
from torch import Tensor

PROMPT_DROPOUT = 0.1


def sample_t(batch: int, shift: float, device: torch.device | None = None) -> Tensor:
    """Logit-normal t. The resolution shift s·t / (1 + (s - 1)·t) is a translation of the logit by ln s."""
    return torch.sigmoid(torch.randn(batch, device=device) + math.log(shift))


def t_quantiles(n: int, shift: float) -> Tensor:
    """The midpoints of `n` equal-probability bins of `sample_t`'s distribution: a fixed t grid that weighs t as training does."""
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.sigmoid(torch.special.ndtri(p) + math.log(shift)).float()


def noisy(x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * x + t * eps


def target(x: Tensor, eps: Tensor, x_t: Tensor) -> Tensor:
    """T = x_t - v with v = ε - x."""
    return x_t - (eps - x)


def per_sample_loss(out: Tensor, x: Tensor, eps: Tensor, x_t: Tensor) -> Tensor:
    return (out - target(x, eps, x_t)).square().flatten(1).mean(1)


def drop_prompts(text: Tensor, null_embedding: Tensor) -> Tensor:
    dropped = torch.rand(text.shape[0], 1, device=text.device) < PROMPT_DROPOUT
    return torch.where(dropped, null_embedding, text)


def flow_loss(model, x: Tensor, text: Tensor, shift: float) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the batch loss, the per-sample losses and the sampled t, for `loss_by_t_bucket`."""
    t = sample_t(x.shape[0], shift, x.device)
    eps = torch.randn_like(x)
    x_t = noisy(x, eps, t)
    out = model(x_t, t, drop_prompts(text, model.null_embedding))
    per_sample = per_sample_loss(out, x, eps, x_t)
    return per_sample.mean(), per_sample, t


def loss_by_t_bucket(per_sample: Tensor, t: Tensor, buckets: int) -> Tensor:
    """Mean loss in each of `buckets` equal-width t bins; NaN where a bin is empty."""
    index = (t * buckets).long().clamp(max=buckets - 1)
    total = torch.zeros(buckets, dtype=per_sample.dtype, device=per_sample.device).index_add_(0, index, per_sample)
    count = torch.zeros_like(total).index_add_(0, index, torch.ones_like(per_sample))
    return total / count
