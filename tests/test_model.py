import copy

import pytest
import torch

from tinydit.model import DIM, DiT, HEAD_DIM, ROPE_DIM, apply_rope, patchify, rope_angles, unpatchify


@pytest.mark.parametrize("size, tokens", [(128, 64), (256, 256)])
def test_output_shape_and_token_count(model, text, size, tokens):
    x = torch.randn(3, 2, size // 2, size // 2)
    assert patchify(x)[0].shape == (3, tokens, 128)
    assert model(x, torch.rand(3), text).shape == x.shape


def test_patchify_roundtrip():
    x = torch.randn(2, 2, 64, 64)
    tokens, h, w = patchify(x)
    assert torch.equal(unpatchify(tokens, h, w), x)


def test_patchify_places_each_latent_pixel_in_its_token_and_channel():
    # Non-square: 4 x 6 tokens. Pixel (channel 1, row 19, col 42) is in patch (2, 5), at offset (3, 2) inside it.
    x = torch.zeros(1, 2, 32, 48)
    x[0, 1, 19, 42] = 1
    tokens, h, w = patchify(x)
    assert (h, w) == (4, 6)
    expected = torch.zeros(1, 24, 128)
    expected[0, 2 * 6 + 5, 1 * 64 + 3 * 8 + 2] = 1
    assert torch.equal(tokens, expected)
    assert torch.equal(unpatchify(tokens, h, w), x)


def test_output_is_exactly_zero_at_init(model, text):
    x = torch.randn(3, 2, 64, 64)
    assert (model(x, torch.rand(3), text) == 0).all()


def test_conv_residual_is_identity_at_init(model):
    y = torch.randn(1, 512, 8, 8)
    assert (model.blocks[0].conv(y) == 0).all()


@pytest.mark.parametrize("row, col", [(1, 4), (3, 0)])
def test_conv_acts_on_the_token_grid_of_a_non_square_image(model, row, col):
    """An impulse at token (row, col) of a 4 x 6 grid reaches exactly its 3 x 3 neighbours, in the QKV input."""
    h, w = 4, 6
    block = model.blocks[0]
    block.table.data.zero_()
    kernel = torch.arange(1.0, 10.0).view(3, 3)
    block.conv.weight.data[:] = kernel
    seen = []
    block.qkv.register_forward_pre_hook(lambda module, args: seen.append(args[0]))

    x = torch.zeros(1, h * w, DIM)
    x[0, row * w + col] = 1  # the unit-RMS vector of ones survives the norm unchanged
    block(x, torch.zeros(1, 6, DIM), rope_angles(h, w), h, w)

    expected = torch.zeros(h, w)
    expected[row, col] = 1  # the residual
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if 0 <= row + dr < h and 0 <= col + dc < w:
                expected[row + dr, col + dc] += kernel[1 - dr, 1 - dc]  # cross-correlation
    assert torch.allclose(seen[0][0], expected.reshape(h * w, 1).expand(h * w, DIM))


def test_gate_rows_start_at_zero_and_shift_scale_rows_do_not(model):
    weight = model.ada[1].weight.view(8, 512, 512)
    table = model.blocks[0].table
    for rows in (weight, table):
        assert (rows[[2, 5]] == 0).all()
        assert (rows[[0, 1, 3, 4]] != 0).any(dim=-1).all()
    assert (weight[[6, 7]] != 0).any(dim=-1).all() and (model.final_table != 0).any(dim=-1).all()  # the final shift, scale


def test_the_final_modulation_starts_near_identity(model, text):
    """The final norm's shift/scale come through the shared projection like the blocks', so its
    gain `1 + scale` starts near 1, not spread by raw `c` (std 1.07, |scale| up to 4; FACTS, adaLN at init)."""
    seen = {}
    model.final_norm.register_forward_hook(lambda module, args, out: seen.update(normed=out))
    model.final.register_forward_pre_hook(lambda module, args: seen.update(modulated=args[0]))
    model(torch.randn(3, 2, 64, 64), torch.rand(3), text)
    n = seen["normed"] - seen["normed"].mean(1, keepdim=True)
    y = seen["modulated"] - seen["modulated"].mean(1, keepdim=True)
    scale = (n * y).sum(1) / n.square().sum(1) - 1  # per sample and channel, the slope over tokens
    assert scale.std() < 0.5 and scale.abs().max() < 2


def test_the_last_two_rows_of_the_shared_projection_modulate_the_final_norm_only(model, text):
    seen = {}
    model.blocks[0].register_forward_pre_hook(lambda module, args: seen.update(block=args[1].clone()))
    model.final.register_forward_pre_hook(lambda module, args: seen.update(final=args[0].clone()))
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    model(x, t, text)
    before = dict(seen)
    with torch.no_grad():
        model.ada[1].weight.view(8, 512, 512)[6:] += 0.1
    model(x, t, text)
    assert torch.equal(seen["block"], before["block"]) and not torch.allclose(seen["final"], before["final"])


def test_the_text_enters_at_unit_rms_whatever_its_scale(model, text):
    torch.nn.init.normal_(model.final.weight, std=0.05)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    base = model(x, t, text)
    assert (model(x, t, 3 * text) - base).norm() < 1e-4 * base.norm()  # equal up to fp32 rounding in the norm


def test_text_and_timestep_condition_the_network_once_trained(model, text):
    # Perturb the final layer so F is nonzero, then check every input reaches F.
    torch.nn.init.normal_(model.final.weight, std=0.05)
    for block in model.blocks:
        torch.nn.init.normal_(block.conv.weight, std=0.1)
        block.table.data[[2, 5]] = 0.5
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    base = model(x, t, text)
    assert base.abs().max() > 0
    assert not torch.allclose(base, model(x, t, text.flip(0)))
    assert not torch.allclose(base, model(x, 1 - t, text))
    assert not torch.allclose(base, model(x.flip(-1), t, text))


def test_bf16_network_takes_fp32_inputs_and_returns_fp32_close_to_the_fp32_network(model, text):
    torch.nn.init.normal_(model.final.weight, std=0.05)
    for block in model.blocks:
        torch.nn.init.normal_(block.conv.weight, std=0.1)
        block.table.data[[2, 5]] = 0.5
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    reference = model(x, t, text)
    out = copy.deepcopy(model).bfloat16()(x, t, text)
    assert out.dtype == torch.float32 and out.shape == x.shape
    assert (out - reference).norm() < 0.02 * reference.norm()  # measured 0.007


def test_conv_mixes_neighbouring_tokens_only(model):
    conv = model.blocks[0].conv
    torch.nn.init.normal_(conv.weight)
    y = torch.zeros(1, 512, 8, 8)
    y[:, :, 4, 4] = 1
    out = conv(y)
    assert (out[:, :, 3:6, 3:6] != 0).any()
    out[:, :, 3:6, 3:6] = 0
    assert (out == 0).all()


def test_gradient_reaches_every_zero_init_role(model, text):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    for _ in range(4):
        loss = (model(x, t, text) - 1).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    block = model.blocks[0]
    assert model.final.weight.abs().max() > 0
    assert block.conv.weight.abs().max() > 0
    assert block.table[[2, 5]].abs().max() > 0
    assert model.ada[1].weight.view(8, 512, 512)[[2, 5]].abs().max() > 0


class TestRope:
    """With the same q and k vector at every token, logits depend on position only through the RoPE half."""

    def logits(self, q_vec, k_vec, h=6, w=6):
        cos, sin = rope_angles(h, w)
        q = apply_rope(q_vec.expand(1, 1, h * w, HEAD_DIM), cos, sin)
        k = apply_rope(k_vec.expand(1, 1, h * w, HEAD_DIM), cos, sin)
        return (q[..., :, None, :] * k[..., None, :, :])

    def test_free_half_is_position_independent(self):
        q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)
        free = self.logits(q, k)[..., ROPE_DIM:].sum(-1)
        assert torch.allclose(free, free[..., :1, :1].expand_as(free), atol=1e-5)

    def test_rope_half_depends_on_relative_position_only(self):
        q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)
        logit = self.logits(q, k)[..., :ROPE_DIM].sum(-1)[0, 0].view(6, 6, 6, 6)  # [r_q, c_q, r_k, c_k]
        assert torch.allclose(logit[0, 0, 1, 2], logit[3, 2, 4, 4], atol=1e-5)
        assert not torch.allclose(logit[0, 0, 1, 2], logit[0, 0, 2, 2], atol=1e-3)
        assert not torch.allclose(logit[0, 0, 1, 2], logit[0, 0, 1, 3], atol=1e-3)

    def test_rows_and_columns_use_separate_dims(self):
        # Row-only displacement leaves the column dims' contribution unchanged, and vice versa.
        q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)
        pairwise = self.logits(q, k)[0, 0].view(6, 6, 6, 6, HEAD_DIM)
        quarter = ROPE_DIM // 4  # row angle dims are [0:q] and [2q:3q], column [q:2q] and [3q:4q]
        col_dims = list(range(quarter, 2 * quarter)) + list(range(3 * quarter, 4 * quarter))
        col_logit = pairwise[..., col_dims].sum(-1)
        assert torch.allclose(col_logit[0, 0, 0, 0], col_logit[0, 0, 3, 0], atol=1e-5)
        assert not torch.allclose(col_logit[0, 0, 0, 0], col_logit[0, 0, 0, 3], atol=1e-3)
