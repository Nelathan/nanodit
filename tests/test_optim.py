import torch

from tinydit.model import DiT
from tinydit.optim import build_optimizers


def test_split_covers_every_parameter_once_and_routes_by_role():
    model = DiT(torch.randn(768), depth=2)
    matrix, fallback = build_optimizers(model, 1e-3, 0.95, 1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    usui = {names[id(p)] for g in matrix.param_groups for p in g["params"]}
    adam = {names[id(p)] for g in fallback.param_groups for p in g["params"]}
    assert usui | adam == set(names.values()) and not usui & adam
    assert all(model.get_parameter(n).ndim == 2 for n in usui)
    assert {"final.weight", "null_embedding", "final_table", "blocks.0.table", "blocks.0.conv.weight", "final.bias"} <= adam
    assert {"ada.1.weight", "blocks.0.qkv.weight", "blocks.0.gate.weight", "blocks.0.out.weight", "patch_embed.weight"} <= usui


def test_residual_stream_output_matrices_track_their_left_side():
    model = DiT(torch.randn(768), depth=2)
    matrix, _ = build_optimizers(model, 1e-3, 0.95, 1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    side = {names[id(p)]: g["side"] for g in matrix.param_groups for p in g["params"]}
    for name in ("blocks.0.out.weight", "blocks.1.w_down.weight", "patch_embed.weight"):
        assert side[name] == "left"
    for name in ("blocks.0.qkv.weight", "blocks.0.w_up.weight", "ada.1.weight"):
        assert side[name] == "auto"
