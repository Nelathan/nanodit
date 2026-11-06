import copy

import pytest
import torch

from conftest import open_gates
from tinydit.model import DiT, HEAD_DIM, N_MOD, Prompt, TOKEN_DIM, apply_rope, patchify, rope_angles, unpatchify


@pytest.mark.parametrize("size, tokens", [(128, 64), (256, 256)])
def test_output_shape_and_token_count(model, prompt, size, tokens):
    x = torch.randn(3, 2, size // 2, size // 2)
    assert patchify(x)[0].shape == (3, tokens, 128)
    assert model(x, torch.rand(3), prompt).shape == x.shape


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


def test_output_is_the_input_at_init(model, prompt):
    """The stream's first 128 channels start as the input patch and the zero gates add nothing, so F = x_t: v = 0."""
    x = torch.randn(3, 2, 64, 64)
    assert torch.equal(model(x, torch.rand(3), prompt), x)


def test_patch_embed_starts_as_the_identity_on_the_first_channels(model):
    assert torch.equal(model.patch_embed.weight[:TOKEN_DIM], torch.eye(TOKEN_DIM))
    assert (model.patch_embed.bias == 0).all()


def test_output_is_the_first_channels_of_the_stream(model, prompt):
    open_gates(model)
    seen = {}
    model.blocks[-1].register_forward_hook(lambda module, args, out: seen.update(stream=out))
    out = model(torch.randn(3, 2, 64, 64), torch.rand(3), prompt)
    assert torch.equal(out, unpatchify(seen["stream"][..., :TOKEN_DIM].float(), 8, 8))


def test_gate_rows_start_at_zero_and_shift_scale_rows_do_not(model):
    weight = model.ada[1].weight.view(N_MOD, 512, 512)
    table = model.blocks[0].table
    for rows in (weight, table):
        assert (rows[[2, 5, 8]] == 0).all()
        assert (rows[[0, 1, 3, 4, 6, 7]] != 0).any(dim=-1).all()


def test_the_text_enters_at_unit_rms_whatever_its_scale(model, prompt):
    open_gates(model)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    base = model(x, t, prompt)
    assert (model(x, t, Prompt(3 * prompt.pooled, 3 * prompt.tokens, prompt.mask)) - base).norm() < 1e-4 * base.norm()  # equal up to fp32 rounding in the norm


def test_text_and_timestep_condition_the_network_once_trained(model, prompt):
    # Open the gates so the blocks reach F, then check every input does.
    open_gates(model)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    base = model(x, t, prompt)
    assert not torch.equal(base, x)
    assert not torch.allclose(base, model(x, t, Prompt(*(a.flip(0) for a in prompt))))
    assert not torch.allclose(base, model(x, 1 - t, prompt))
    assert not torch.allclose(base, model(x.flip(-1), t, prompt))


def test_bf16_network_takes_fp32_inputs_and_returns_fp32_close_to_the_fp32_network(model, prompt):
    open_gates(model)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    reference = model(x, t, prompt)
    out = copy.deepcopy(model).bfloat16()(x, t, prompt)
    assert out.dtype == torch.float32 and out.shape == x.shape
    assert (out - reference).norm() < 0.02 * reference.norm()  # measured 0.007


def test_gradient_reaches_every_zero_init_role(model, prompt):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    for _ in range(4):
        loss = (model(x, t, prompt) - 1).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    block = model.blocks[0]
    assert block.table[[2, 5, 8]].abs().max() > 0
    assert model.ada[1].weight.view(N_MOD, 512, 512)[[2, 5, 8]].abs().max() > 0
    assert block.qkv.weight.grad.abs().max() > 0  # behind the gates, reached once they opened
    assert block.cross_kv.weight.grad.abs().max() > 0


def test_the_token_states_condition_the_network_and_padding_does_not(model, prompt):
    open_gates(model)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    base = model(x, t, prompt)
    assert not torch.allclose(base, model(x, t, prompt._replace(tokens=prompt.tokens.flip(0))))
    noise = torch.randn_like(prompt.tokens) * ~prompt.mask[..., None]  # garbage where the mask is false
    assert torch.equal(base, model(x, t, prompt._replace(tokens=prompt.tokens + noise)))


def test_a_caption_is_the_same_alone_and_padded_in_a_batch(model, prompt):
    open_gates(model)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    batched = model(x, t, prompt)
    n = int(prompt.mask[2].sum())
    alone = Prompt(prompt.pooled[2:], prompt.tokens[2:, :n], prompt.mask[2:, :n])
    assert torch.allclose(model(x[2:], t[2:], alone), batched[2:], atol=1e-5)


class TestRope:
    """With the same q and k vector at every token, logits depend on position only through RoPE."""

    def logits(self, q_vec, k_vec, h=6, w=6):
        cos, sin = rope_angles(h, w)
        q = apply_rope(q_vec.expand(1, 1, h * w, HEAD_DIM), cos, sin)
        k = apply_rope(k_vec.expand(1, 1, h * w, HEAD_DIM), cos, sin)
        return (q[..., :, None, :] * k[..., None, :, :])

    def test_logits_depend_on_relative_position_only(self):
        q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)
        logit = self.logits(q, k).sum(-1)[0, 0].view(6, 6, 6, 6)  # [r_q, c_q, r_k, c_k]
        assert torch.allclose(logit[0, 0, 1, 2], logit[3, 2, 4, 4], atol=1e-5)
        assert not torch.allclose(logit[0, 0, 1, 2], logit[0, 0, 2, 2], atol=1e-3)
        assert not torch.allclose(logit[0, 0, 1, 2], logit[0, 0, 1, 3], atol=1e-3)

    def test_rows_and_columns_use_separate_dims(self):
        # Row-only displacement leaves the column dims' contribution unchanged, and vice versa.
        q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)
        pairwise = self.logits(q, k)[0, 0].view(6, 6, 6, 6, HEAD_DIM)
        quarter = HEAD_DIM // 4  # row angle dims are [0:q] and [2q:3q], column [q:2q] and [3q:4q]
        col_dims = list(range(quarter, 2 * quarter)) + list(range(3 * quarter, 4 * quarter))
        col_logit = pairwise[..., col_dims].sum(-1)
        assert torch.allclose(col_logit[0, 0, 0, 0], col_logit[0, 0, 3, 0], atol=1e-5)
        assert not torch.allclose(col_logit[0, 0, 0, 0], col_logit[0, 0, 0, 3], atol=1e-3)


def test_a_branch_adds_at_most_its_gate_however_large_its_weights(model, prompt):
    open_gates(model)
    x, t = torch.randn(3, 2, 64, 64), torch.rand(3)
    base = model(x, t, prompt)
    for block in model.blocks:
        block.w_down.weight.data *= 1e3
        block.out.weight.data *= 1e3
        block.cross_out.weight.data *= 1e3
    assert torch.allclose(model(x, t, prompt), base, atol=1e-3)  # the post-norm removes the weights' scale
