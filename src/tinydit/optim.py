"""Optimizers: UsuiTrack for the 2-D Linear weights, StochasticAdamW for the rest (SPEC §5, §6)."""

from torch import nn
from usuitrack import Molegrad, StochasticAdamW, UsuiTrack

# Weights on the residual stream's output side track their output (left) side; the rest read it or
# have no stream side, and follow Usui's shape rule. Same convention as ai-toolkit's usuitrack.py.
LEFT_SIDE = ("out.weight", "w_down.weight", "patch_embed.weight")
# Tables are not shared linear maps.
USUI_EXCLUDED = ("table", "final")


def build_optimizers(model: nn.Module, usui_lr: float, beta: float, rank_fraction: float, fallback_lr: float, pin_weights: bool = False) -> tuple[UsuiTrack, StochasticAdamW]:
    """UsuiTrack for the 2-D Linear weights, StochasticAdamW (AdamW whose bf16 updates round stochastically)
    for everything else (1-D, biases, tables, null embedding); betas (0.9, 0.99), no
    weight decay, as in usuitrack SPEC "Fallback path".
    Molegrad is attached to the model (which must already be on its device), so the tracked matrices
    never get a `.grad`; it hooks the optimizer, which stays the only handle.
    Every tracked matrix takes Usui's fixed step, one lr unit for all, from a frame drawn at random (SPEC §6).
    With `pin_weights` each keeps its init norm and takes the trust step instead, so `usui_lr` is a constant relative step."""
    sided, fallback = {"left": [], "auto": []}, []
    for name, param in model.named_parameters():
        if param.ndim == 2 and not any(key in name for key in USUI_EXCLUDED):
            sided["left" if name.endswith(LEFT_SIDE) else "auto"].append(param)
        else:
            fallback.append(param)
    groups = [{"params": params, "side": side, "step_rule": "trust" if pin_weights else "fixed", "pin_norm": pin_weights} for side, params in sided.items()]
    matrix = UsuiTrack(groups, lr=usui_lr, beta=beta, rank_fraction=rank_fraction, frame_init="random", diagnostics=True)
    Molegrad(model, matrix)
    return matrix, StochasticAdamW(fallback, lr=fallback_lr, betas=(0.9, 0.99), weight_decay=0.0)
