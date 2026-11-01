"""Optimizers: UsuiTrack for the 2-D Linear weights, StochasticAdamW for the rest (SPEC §5, §6)."""

from torch import nn
from usuitrack import Molegrad, StochasticAdamW, UsuiTrack

# Weights on the residual stream's output side track their output (left) side; the rest read it or
# have no stream side, and follow Usui's shape rule. Same convention as ai-toolkit's usuitrack.py.
LEFT_SIDE = ("out.weight", "w_down.weight", "patch_embed.weight")
# Tables are not shared linear maps.
USUI_EXCLUDED = ("table",)
# The zero-initialized final linear builds its whole map from Usui's steps: it takes the largest frame,
# half its shorter side (r = 64 for [128, 512]); a tangent needs directions outside the frame.
FINAL_RANK_FRACTION = 1.0


def build_optimizers(model: nn.Module, usui_lr: float, beta: float, rank_fraction: float, fallback_lr: float) -> tuple[UsuiTrack, StochasticAdamW]:
    """UsuiTrack for the 2-D Linear weights, StochasticAdamW (AdamW whose bf16 updates round stochastically)
    for everything else (1-D, biases, conv, tables, null embedding); betas (0.9, 0.99), no weight decay,
    as in usuitrack SPEC "Fallback path".
    Molegrad is attached to the model (which must already be on its device), so the tracked matrices
    never get a `.grad`; it hooks the optimizer, which stays the only handle.
    Every matrix takes Usui's fixed step: trained from scratch, the zero-initialized final linear has no
    norm to step relative to, and one lr unit holds for all (SPEC §6)."""
    sided, fallback = {"left": [], "auto": []}, []
    for name, param in model.named_parameters():
        if name == "final.weight":
            continue
        if param.ndim == 2 and not any(key in name for key in USUI_EXCLUDED):
            sided["left" if name.endswith(LEFT_SIDE) else "auto"].append(param)
        else:
            fallback.append(param)
    groups = [{"params": params, "side": side, "step_rule": "fixed"} for side, params in sided.items()]
    groups.append({"params": [model.final.weight], "side": "auto", "step_rule": "fixed", "rank_fraction": FINAL_RANK_FRACTION})
    matrix = UsuiTrack(groups, lr=usui_lr, beta=beta, rank_fraction=rank_fraction, diagnostics=True)
    Molegrad(model, matrix)
    return matrix, StochasticAdamW(fallback, lr=fallback_lr, betas=(0.9, 0.99), weight_decay=0.0)
