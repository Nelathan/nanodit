"""Optimizers. PROVISIONAL: this split is option (a) of PLAN A1, pending Daniel's decision."""

import torch
from torch import nn
from usuitrack import Molegrad, UsuiTrack

# Weights on the residual stream's output side track their output (left) side; the rest read it or
# have no stream side, and follow Usui's shape rule. Same convention as ai-toolkit's usuitrack.py.
LEFT_SIDE = ("out.weight", "w_down.weight", "patch_embed.weight")
# UsuiTrack never moves a zero-norm matrix (usuitrack SPEC "Limit"), and tables are not shared linear maps.
USUI_EXCLUDED = ("final.weight", "table")


def build_optimizers(model: nn.Module, usui_lr: float, beta: float, fallback_lr: float) -> tuple[UsuiTrack, torch.optim.AdamW]:
    """UsuiTrack for the 2-D Linear weights, AdamW for everything else (1-D, biases, conv, tables,
    null embedding, final linear); betas (0.9, 0.99), no weight decay, as in usuitrack SPEC "Fallback path".
    Molegrad is attached to the model (which must already be on its device), so the tracked matrices
    never get a `.grad`; it hooks the optimizer, which stays the only handle."""
    sided, fallback = {"left": [], "auto": []}, []
    for name, param in model.named_parameters():
        if param.ndim == 2 and not any(key in name for key in USUI_EXCLUDED):
            sided["left" if name.endswith(LEFT_SIDE) else "auto"].append(param)
        else:
            fallback.append(param)
    matrix = UsuiTrack([{"params": params, "side": side} for side, params in sided.items()], lr=usui_lr, beta=beta, diagnostics=True)
    Molegrad(model, matrix)
    return matrix, torch.optim.AdamW(fallback, lr=fallback_lr, betas=(0.9, 0.99), weight_decay=0.0)
