"""SPEC §4: x0 velocity on rectified flow, x_t = (1 - t)·x + t·ε."""

import math

import torch
from torch import Tensor

from .model import Prompt

PROMPT_DROPOUT = 0.1
DECODER_GAIN_EXPONENT = 1.52  # the decoder's gain on a latent error falls as f^-1.52 (log 2026-11-22-decoder-gain-power-law)
GAIN_FLOOR = 1 / 64  # cycles per latent pixel; below it the gain is held


def sample_t(batch: int, shift: float, device: torch.device | None = None, sigma: float = 1.0) -> Tensor:
    """Logit-normal t with logit std `sigma`. The resolution shift s·t / (1 + (s - 1)·t) is a translation of the logit by ln s."""
    return torch.sigmoid(sigma * torch.randn(batch, device=device) + math.log(shift))


def t_quantiles(n: int, shift: float) -> Tensor:
    """The midpoints of `n` equal-probability bins of the unit-sigma `sample_t`: a fixed t grid."""
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.sigmoid(torch.special.ndtri(p) + math.log(shift)).float()


def t_uniform(n: int) -> Tensor:
    """The midpoints of `n` equal bins of t: a fixed grid that weights t as the sampler's evenly spaced steps do."""
    return (torch.arange(n, dtype=torch.float64) + 0.5).div(n).float()


def noisy(x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    t = t.view(-1, *[1] * (x.dim() - 1))
    return (1 - t) * x + t * eps


def target(x: Tensor, eps: Tensor, x_t: Tensor) -> Tensor:
    """T = x_t - v with v = ε - x."""
    return x_t - (eps - x)


# Spatial-frequency bands of the 64 x 64 latent by the period in tokens (8 latent px): more than three tokens, beyond a
# 3x3 conv's reach; one to three; inside one token. Edges in radial cycles per latent pixel; a period of exactly one or
# three tokens falls in the coarser band.
BAND_EDGES = (1 / 24, 1 / 8)
BAND_NAMES = ("over3tok", "1to3tok", "in1tok")


def band_energy(e: Tensor) -> Tensor:
    """Mean square of e [B, C, H, W] in each band, summed over bands equal to the mean square of e: [B, len(BAND_NAMES)]."""
    H, W = e.shape[-2:]
    spectrum = torch.fft.fft2(e).abs().square().sum(1) / (H * W) ** 2 / e.shape[1]  # Parseval: sums to mean square
    fy, fx = torch.fft.fftfreq(H, device=e.device), torch.fft.fftfreq(W, device=e.device)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    band = torch.bucketize(radius, torch.tensor(BAND_EDGES, device=e.device))  # edges[i - 1] < radius <= edges[i]
    return torch.stack([spectrum[:, band == i].sum(-1) for i in range(len(BAND_NAMES))], -1)


def spectral_weight(h: int, w: int, alpha: float, device: torch.device | None = None) -> Tensor:
    """w(f)² [h, w] on the fft2 grid: the decoder's gain to the power alpha, max(f, GAIN_FLOOR)^(-alpha·β), with
    mean 1 over the grid, so a white error keeps its loss. alpha 0 is the plain MSE, 1 about the image's MSE."""
    fy, fx = torch.fft.fftfreq(h, device=device), torch.fft.fftfreq(w, device=device)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt().clamp(min=GAIN_FLOOR)
    weight = radius.pow(-alpha * DECODER_GAIN_EXPONENT)
    return weight / weight.mean()


def error_power(out: Tensor, x: Tensor, eps: Tensor, x_t: Tensor) -> Tensor:
    """|fft2(F - T)|² [B, C, H, W], orthonormal: its mean is the error's mean square (Parseval)."""
    return torch.fft.fft2((out - target(x, eps, x_t)).float(), norm="ortho").abs().square()


def weighted_mean(power: Tensor, alpha: float) -> Tensor:
    """The per-sample mean of `error_power`, each spatial frequency weighted by `spectral_weight`; any fixed weight
    keeps E[T | x_t] as the minimizer."""
    return (power * spectral_weight(*power.shape[-2:], alpha, power.device)).flatten(1).mean(1)


def per_sample_loss(out: Tensor, x: Tensor, eps: Tensor, x_t: Tensor, alpha: float = 0.0) -> Tensor:
    return weighted_mean(error_power(out, x, eps, x_t), alpha)


def drop_prompts(prompt: Prompt, null: Prompt) -> Prompt:
    """Each row becomes the `null` prompt with probability PROMPT_DROPOUT."""
    dropped = torch.rand(prompt.mask.shape[0], device=prompt.mask.device) < PROMPT_DROPOUT
    return Prompt(*(torch.where(dropped.view(-1, *[1] * (a.dim() - 1)), b, a) for a, b in zip(prompt, null)))


def flow_loss(model, x: Tensor, prompt: Prompt, shift: float, sigma: float = 1.0, alpha: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the batch loss it trains on, weighted by `alpha`, and the per-sample unweighted losses and the sampled
    t, for `loss_by_t_bucket` and `explained_variance`, which read the plain MSE whatever alpha."""
    t = sample_t(x.shape[0], shift, x.device, sigma)
    eps = torch.randn_like(x)
    x_t = noisy(x, eps, t)
    prompt = drop_prompts(prompt, model.null_prompt(*prompt.tokens.shape[:2]))
    power = error_power(model(x_t, t, prompt), x, eps, x_t)
    return weighted_mean(power, alpha).mean(), power.detach().flatten(1).mean(1), t


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
