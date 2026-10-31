import copy

import torch

from tinydit.model import DiT
from tinydit.objective import flow_loss, noisy, target
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


def trained_model(steps=2):
    """A depth-2 model with a nonzero final layer (so every matrix gets a gradient), stepped `steps` times."""
    torch.manual_seed(0)
    model = DiT(torch.randn(768), depth=2)
    torch.nn.init.normal_(model.final.weight, std=0.05)
    optimizers = build_optimizers(model, 3e-3, 0.95, 1e-3)
    for _ in range(steps):
        train_step(model, optimizers, *batch(0))
    return model, optimizers


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 2, 32, 32, generator=g)
    return x, torch.nn.functional.normalize(torch.randn(8, 768, generator=g), dim=-1)


def train_step(model, optimizers, x, text, micro_batch=8):
    for i in range(0, len(x), micro_batch):
        loss, _, _ = flow_loss(model, x[i : i + micro_batch], text[i : i + micro_batch], shift=1.0)
        (loss * micro_batch / len(x)).backward()
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad()


def test_tracked_matrices_have_no_gradient_and_still_move():
    model, optimizers = trained_model(steps=0)
    qkv = model.blocks[0].qkv.weight
    before = qkv.detach().clone()
    train_step(model, optimizers, *batch(0))
    train_step(model, optimizers, *batch(1))
    assert qkv.grad is None and not qkv.requires_grad
    assert not torch.equal(qkv, before)
    assert model.final.weight.requires_grad  # fallback parameters keep their gradients


def test_accumulated_micro_batches_give_the_full_batch_fallback_gradient():
    x, text = batch(0)
    grads = []
    for micro_batch in (8, 2):
        torch.manual_seed(1)
        model = DiT(torch.randn(768), depth=2)
        torch.nn.init.normal_(model.final.weight, std=0.05)
        build_optimizers(model, 3e-3, 0.95, 1e-3)
        torch.manual_seed(2)  # t, ε and dropout draws differ with the split, so fix the inputs instead
        t, eps = torch.rand(8), torch.randn_like(x)
        for i in range(0, 8, micro_batch):
            sl = slice(i, i + micro_batch)
            x_t = noisy(x[sl], eps[sl], t[sl])
            loss = (model(x_t, t[sl], text[sl]) - target(x[sl], eps[sl], x_t)).square().mean()
            (loss * micro_batch / 8).backward()
        grads.append(model.final.weight.grad.clone())
    assert torch.allclose(*grads, atol=1e-6)


def test_checkpoint_round_trip_continues_exactly_where_training_left_off():
    model, optimizers = trained_model()
    state = copy.deepcopy((model.state_dict(), [o.state_dict() for o in optimizers]))  # state_dict() aliases live tensors
    torch.manual_seed(5)
    train_step(model, optimizers, *batch(1))  # the uninterrupted run

    restored = DiT(torch.zeros(768), depth=2)
    restored.load_state_dict(state[0])
    resumed = build_optimizers(restored, 3e-3, 0.95, 1e-3)
    for optimizer, saved in zip(resumed, state[1]):
        optimizer.load_state_dict(saved)
    qkv = restored.blocks[0].qkv
    assert not qkv.basis.requires_grad  # reads the loaded frame, no refit pending
    assert torch.equal(qkv.basis, resumed[0].state[qkv.weight]["frame"])
    torch.manual_seed(5)
    train_step(restored, resumed, *batch(1))
    for (name, a), (_, b) in zip(model.named_parameters(), restored.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name
