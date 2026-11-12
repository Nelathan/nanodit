import copy

import torch
from conftest import PAIR, open_gates

from nanodit.data import pad_sequences
from nanodit.model import NULL_TOKENS, SEQ_DIM, TOKEN_DIM, DiT, Prompt
from nanodit.objective import flow_loss, noisy, target
from nanodit.optim import build_optimizers


def test_split_covers_every_parameter_once_and_routes_by_role():
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR)
    matrix, fallback = build_optimizers(model, 1e-3, 0.95, 0.1, 1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    usui = {names[id(p)] for g in matrix.param_groups for p in g["params"]}
    adam = {names[id(p)] for g in fallback.param_groups for p in g["params"]}
    assert usui | adam == set(names.values()) and not usui & adam
    assert all(model.get_parameter(n).ndim == 2 for n in usui)
    assert {"null_embedding", "blocks.0.table", "patch_embed.bias", "blocks.1.q_norm.weight", "blocks.1.attn_norm.weight", "blocks.0.conv.weight", "blocks.0.conv_norm.weight", "blocks.1.mlp_norm.weight"} <= adam
    assert {"ada.1.weight", "blocks.1.qkv.weight", "blocks.1.gate.weight", "blocks.0.out.weight", "patch_embed.weight"} <= usui
    assert {(g["step_rule"], g["frame_init"], g["norm_bounds"]) for g in matrix.param_groups} == {("trust", "random", (None, 1.0))}


def test_residual_stream_output_matrices_track_their_left_side():
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR)
    matrix, _ = build_optimizers(model, 1e-3, 0.95, 0.1, 1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    side = {names[id(p)]: g["side"] for g in matrix.param_groups for p in g["params"]}
    for name in ("blocks.0.out.weight", "blocks.1.w_down.weight", "patch_embed.weight"):
        assert side[name] == "left"
    for name in ("blocks.1.qkv.weight", "blocks.0.w_up.weight", "ada.1.weight"):
        assert side[name] == "auto"


def test_the_first_step_moves_only_what_the_zero_gates_leave_a_gradient():
    torch.manual_seed(0)
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR).bfloat16()
    fallback_lr = 1e-3
    optimizers = build_optimizers(model, 3e-3, 0.95, 0.1, fallback_lr)
    tracked = {id(p) for g in optimizers[0].param_groups for p in g["params"]}
    matrices = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) in tracked}
    train_step(model, optimizers, *batch(0))
    change = {n: ((model.get_parameter(n).float() - before.float()).norm() / before.float().norm()).item() for n, before in matrices.items()}
    # F = x_t at init: the loss reaches the shared projection through the gate rows, and nothing else, since the
    # patch embed fills meta channels that no gate lets into F yet; every other tracked matrix sits behind a zero
    # gate: its gradient is exactly zero, so its random frame catches nothing and it does not move
    framed = {n for n, p in model.named_parameters() if "frame" in optimizers[0].state.get(p, {})}
    assert len(matrices) > 10 and framed == set(matrices)
    assert change["ada.1.weight"] > 0 and all(c == 0 for n, c in change.items() if n != "ada.1.weight")
    # AdamW's first step is lr times the gradient's sign, up to bf16 rounding
    # (only the first 128 channels, the ones F reads, have an upstream gradient yet)
    assert abs(model.blocks[1].table[[2, 5], :TOKEN_DIM].float().abs().mean() / fallback_lr - 1) < 0.05


def trained_model(steps=2):
    """A depth-2 bf16 model with open gates (so every matrix gets a gradient), stepped `steps` times."""
    torch.manual_seed(0)
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR)
    open_gates(model)
    model.bfloat16()
    optimizers = build_optimizers(model, 3e-3, 0.95, 0.1, 1e-3)
    for _ in range(steps):
        train_step(model, optimizers, *batch(0))
    return model, optimizers


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 2, 32, 32, generator=g)
    tokens, mask = pad_sequences([torch.randn(2 + i % 4, SEQ_DIM, generator=g) for i in range(8)], torch.float32)
    return x, Prompt(torch.nn.functional.normalize(torch.randn(8, 768, generator=g), dim=-1), tokens, mask)


def train_step(model, optimizers, x, prompt, micro_batch=8):
    for i in range(0, len(x), micro_batch):
        loss, *_ = flow_loss(model, x[i : i + micro_batch], prompt.rows(i, i + micro_batch), shift=1.0)
        (loss * micro_batch / len(x)).backward()
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad()


def test_tracked_matrices_have_no_gradient_and_still_move():
    model, optimizers = trained_model(steps=0)
    qkv = model.blocks[1].qkv.weight
    before = qkv.detach().clone()
    train_step(model, optimizers, *batch(0))
    train_step(model, optimizers, *batch(1))
    assert qkv.grad is None and not qkv.requires_grad
    assert not torch.equal(qkv, before)
    assert optimizers[0].state[qkv]["frame"].shape[-1] == round(0.1 * (768 * 512) ** 0.5)  # rank_fraction 0.1
    assert model.blocks[0].table.requires_grad  # fallback parameters keep their gradients


def test_the_production_layout_trains_a_step():
    torch.manual_seed(0)
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=4, width=512).bfloat16()
    open_gates(model)
    optimizers = build_optimizers(model, 3e-3, 0.95, 0.1, 1e-3)
    conv, cross = model.blocks[0].conv.weight.detach().clone(), model.blocks[3].cross_kv.weight.detach().clone()
    train_step(model, optimizers, *batch(0))
    assert not torch.equal(model.blocks[0].conv.weight, conv) and not torch.equal(model.blocks[3].cross_kv.weight, cross)


def test_accumulated_micro_batches_give_the_full_batch_fallback_gradient():
    x, prompt = batch(0)
    grads = []
    for micro_batch in (8, 2):
        torch.manual_seed(1)
        model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR)
        open_gates(model)
        build_optimizers(model.bfloat16(), 3e-3, 0.95, 0.1, 1e-3)
        torch.manual_seed(2)  # t, ε and dropout draws differ with the split, so fix the inputs instead
        t, eps = torch.rand(8), torch.randn_like(x)
        for i in range(0, 8, micro_batch):
            sl = slice(i, i + micro_batch)
            x_t = noisy(x[sl], eps[sl], t[sl])
            loss = (model(x_t, t[sl], prompt.rows(i, i + micro_batch)) - target(x[sl], eps[sl], x_t)).square().mean()
            (loss * micro_batch / 8).backward()
        grads.append(model.blocks[0].table.grad.clone())
    # bf16 sums the split batch in a different order, so equal only up to rounding; a lost or doubled
    # micro-batch would be off by tens of percent
    assert grads[0].abs().max() > 0 and (grads[0].float() - grads[1].float()).norm() < 0.03 * grads[0].float().norm()  # measured 0.003


def test_checkpoint_round_trip_continues_exactly_where_training_left_off():
    model, optimizers = trained_model()
    state = copy.deepcopy((model.state_dict(), [o.state_dict() for o in optimizers]))  # state_dict() aliases live tensors
    torch.manual_seed(5)
    train_step(model, optimizers, *batch(1))  # the uninterrupted run

    restored = DiT(torch.zeros(768), torch.zeros(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR).bfloat16()
    restored.load_state_dict(state[0])
    resumed = build_optimizers(restored, 3e-3, 0.95, 0.1, 1e-3)
    for optimizer, saved in zip(resumed, state[1]):
        optimizer.load_state_dict(saved)
    qkv = restored.blocks[1].qkv
    assert not qkv.basis.requires_grad  # reads the loaded frame, no refit pending
    assert torch.equal(qkv.basis, resumed[0].state[qkv.weight]["frame"])
    torch.manual_seed(5)
    train_step(restored, resumed, *batch(1))
    for (name, a), (_, b) in zip(model.named_parameters(), restored.named_parameters()):
        assert torch.equal(a, b), name  # the stochastic rounding draws repeat under the same seed


def test_no_tracked_norm_grows_past_its_init_norm():
    torch.manual_seed(0)
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR)
    open_gates(model)
    model.bfloat16()
    optimizers = build_optimizers(model, 5e-2, 0.95, 0.1, 1e-3)
    tracked = [p for g in optimizers[0].param_groups for p in g["params"]]
    start = [p.detach().float().norm().item() for p in tracked]
    for seed in range(20):
        train_step(model, optimizers, *batch(seed))
    # unbounded, these 20 steps grow every tracked norm by 21 to 37 %; bf16 rounds the clipped norm by up to 2e-3
    assert all(p.detach().float().norm().item() / norm < 1 + 2e-3 for p, norm in zip(tracked, start, strict=True))
