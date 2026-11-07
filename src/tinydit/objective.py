"""SPEC §4: x0 velocity on rectified flow, x_t = (1 - t)·x + t·ε."""

import math

import torch
from torch import Tensor

from .model import PATCH, Prompt

PROMPT_DROPOUT = 0.1


def sample_t(batch: int, shift: float, device: torch.device | None = None, sigma: float = 1.0) -> Tensor:
    """Logit-normal t with logit std `sigma`. The resolution shift s·t / (1 + (s - 1)·t) is a translation of the logit by ln s."""
    return torch.sigmoid(sigma * torch.randn(batch, device=device) + math.log(shift))


def t_quantiles(n: int, shift: float) -> Tensor:
    """The midpoints of `n` equal-probability bins of the unit-sigma `sample_t`: a fixed t grid."""
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.sigmoid(torch.special.ndtri(p) + math.log(shift)).float()


def noisy(x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * x + t * eps


def masked_tokens(x: Tensor, ratio: float) -> Tensor:
    """[B, 1, H, W] bool, constant on each token's PATCH x PATCH block: each token is masked with probability `ratio`.
    Drawn at ratio 0 too, so masked and unmasked runs of one seed take the same random stream."""
    mask = torch.rand(x.shape[0], 1, x.shape[2] // PATCH, x.shape[3] // PATCH, device=x.device) < ratio
    return mask.repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)


def target(x: Tensor, eps: Tensor, x_t: Tensor) -> Tensor:
    """T = x_t - v with v = ε - x."""
    return x_t - (eps - x)


def per_sample_loss(out: Tensor, x: Tensor, eps: Tensor, x_t: Tensor) -> Tensor:
    return (out - target(x, eps, x_t)).square().flatten(1).mean(1)


def drop_prompts(prompt: Prompt, null: Prompt) -> Prompt:
    """Each row becomes the `null` prompt with probability PROMPT_DROPOUT."""
    dropped = torch.rand(prompt.pooled.shape[0], device=prompt.pooled.device) < PROMPT_DROPOUT
    return Prompt(*(torch.where(dropped.view(-1, *[1] * (a.dim() - 1)), b, a) for a, b in zip(prompt, null)))


def flow_loss(model, x: Tensor, prompt: Prompt, shift: float, sigma: float = 1.0, mask_ratio: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the batch loss, the per-sample losses and the sampled t, for `loss_by_t_bucket`. A share `mask_ratio`
    of the tokens is pure noise, its own t = 1, so its target is the clean latent (masked priming, topics/priming.md);
    the model is told the sample's t, the one of the other tokens."""
    t = sample_t(x.shape[0], shift, x.device, sigma)
    eps = torch.randn_like(x)
    x_t = torch.where(masked_tokens(x, mask_ratio), eps, noisy(x, eps, t))
    out = model(x_t, t, drop_prompts(prompt, model.null_prompt(*prompt.tokens.shape[:2])))
    per_sample = per_sample_loss(out, x, eps, x_t)
    return per_sample.mean(), per_sample, t


def loss_by_t_bucket(per_sample: Tensor, t: Tensor, buckets: int) -> Tensor:
    """Mean loss in each of `buckets` equal-width t bins; NaN where a bin is empty."""
    index = (t * buckets).long().clamp(max=buckets - 1)
    total = torch.zeros(buckets, dtype=per_sample.dtype, device=per_sample.device).index_add_(0, index, per_sample)
    count = torch.zeros_like(total).index_add_(0, index, torch.ones_like(per_sample))
    return total / count


def explained_variance(per_sample: Tensor, t: Tensor, energy: Tensor) -> Tensor:
    """R²: the share of the clean latents' energy `energy` (mean square per sample) that the prediction recovers.
    The F loss is the clean-latent error over t, `per_sample = MSE_x / t²`, so `MSE_x = per_sample · t²`. 0 is
    no better than predicting zero, the prior mean; 1 is exact."""
    return 1 - (per_sample * t.square()).sum() / energy.sum()


def explained_variance_by_t_bucket(per_sample: Tensor, t: Tensor, energy: Tensor, buckets: int) -> Tensor:
    """`explained_variance` in each of `buckets` equal-width t bins; NaN where a bin is empty."""
    index = (t * buckets).long().clamp(max=buckets - 1)
    error = torch.zeros(buckets, dtype=per_sample.dtype, device=per_sample.device).index_add_(0, index, per_sample * t.square())
    return 1 - error / torch.zeros_like(error).index_add_(0, index, energy)
