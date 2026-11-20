"""Optimizers: UsuiTrack for the 2-D Linear weights, StochasticAdamW for the rest (SPEC §5, §6)."""

from torch import nn
from usuitrack import Molegrad, StochasticAdamW, UsuiTrack

# Tables, the null prompt's tokens and the attention sink's null keys are not shared linear maps.
USUI_EXCLUDED = ("table", "null")


def build_optimizers(model: nn.Module, usui_lr: float, beta: float, rank_fraction: float, fallback_lr: float) -> tuple[UsuiTrack, StochasticAdamW]:
    """UsuiTrack for the 2-D Linear weights, StochasticAdamW (AdamW whose bf16 updates round stochastically)
    for everything else (1-D, biases, tables, null tokens), at its defaults but without weight decay.
    Molegrad is attached to the model (which must already be on its device), so the tracked matrices
    never get a `.grad`; it hooks the optimizer, which stays the only handle.
    Usui's defaults otherwise: the trust step, so `usui_lr` is the relative step, every matrix on its input side.
    Frames are drawn at random, and each matrix's norm is clipped at its init norm: it lives in the ball of its init
    radius (SPEC §6)."""
    tracked, fallback = [], []
    for name, param in model.named_parameters():
        (tracked if param.ndim == 2 and not any(key in name for key in USUI_EXCLUDED) else fallback).append(param)
    matrix = UsuiTrack([{"params": tracked, "frame_init": "random", "norm_bounds": (None, 1.0)}], lr=usui_lr, beta=beta, rank_fraction=rank_fraction, diagnostics=True)
    Molegrad(model, matrix)
    return matrix, StochasticAdamW(fallback, lr=fallback_lr, weight_decay=0.0)
