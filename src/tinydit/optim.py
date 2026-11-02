"""Optimizers: UsuiTrack for the 2-D Linear weights but the final linear, StochasticAdamW for the rest (SPEC §5, §6)."""

from torch import nn
from usuitrack import Molegrad, StochasticAdamW, UsuiTrack

# Weights on the residual stream's output side track their output (left) side; the rest read it or
# have no stream side, and follow Usui's shape rule. Same convention as ai-toolkit's usuitrack.py.
LEFT_SIDE = ("out.weight", "w_down.weight", "patch_embed.weight")
# Tables are not shared linear maps.
USUI_EXCLUDED = ("table",)
# The zero-initialized map every output passes through adapts faster per element on the fallback than through a frame.
FINAL_LINEAR = "final.weight"


def build_optimizers(model: nn.Module, usui_lr: float, beta: float, rank_fraction: float, fallback_lr: float) -> tuple[UsuiTrack, StochasticAdamW]:
    """UsuiTrack for the 2-D Linear weights, StochasticAdamW (AdamW whose bf16 updates round stochastically)
    for everything else (the final linear, 1-D, biases, conv, tables, null embedding); betas (0.9, 0.99), no
    weight decay, as in usuitrack SPEC "Fallback path".
    Molegrad is attached to the model (which must already be on its device), so the tracked matrices
    never get a `.grad`; it hooks the optimizer, which stays the only handle.
    Every tracked matrix takes Usui's fixed step, one lr unit for all (SPEC §6)."""
    sided, fallback = {"left": [], "auto": []}, []
    for name, param in model.named_parameters():
        if param.ndim == 2 and name != FINAL_LINEAR and not any(key in name for key in USUI_EXCLUDED):
            sided["left" if name.endswith(LEFT_SIDE) else "auto"].append(param)
        else:
            fallback.append(param)
    groups = [{"params": params, "side": side, "step_rule": "fixed"} for side, params in sided.items()]
    matrix = UsuiTrack(groups, lr=usui_lr, beta=beta, rank_fraction=rank_fraction, diagnostics=True)
    Molegrad(model, matrix)
    return matrix, StochasticAdamW(fallback, lr=fallback_lr, betas=(0.9, 0.99), weight_decay=0.0)
