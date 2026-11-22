import math

import pytest
import torch
from conftest import needs_cuda, open_gates

from nanodit.model import Prompt
from nanodit.objective import PROMPT_DROPOUT, band_energy, drop_prompts, explained_variance, explained_variance_by_t_bucket, flow_loss, loss_by_t_bucket, noisy, per_sample_loss, sample_t, spectral_weight, t_quantiles, target


def test_loss_equals_velocity_error():
    x, eps, t = torch.randn(4, 2, 16, 16), torch.randn(4, 2, 16, 16), torch.rand(4)
    x_t = noisy(x, eps, t)
    out = torch.randn_like(x)
    v = eps - x
    v_hat = x_t - out
    assert torch.allclose((out - target(x, eps, x_t)).square(), (v_hat - v).square(), atol=1e-5)


def test_target_matches_closed_form():
    x, eps, t = torch.randn(4, 2, 16, 16), torch.randn(4, 2, 16, 16), torch.rand(4)
    expected = x + (1 - t.view(-1, 1, 1, 1)) * (x - eps)
    assert torch.allclose(target(x, eps, noisy(x, eps, t)), expected, atol=1e-6)


def stub_null(batch, length):
    """A null prompt for networks without tokens: tokens of -1, only the first token real."""
    return Prompt(-torch.ones(batch, length, 8), (torch.arange(length) < 1).expand(batch, -1))


def test_prompt_dropout_rate():
    torch.manual_seed(0)
    prompt = Prompt(torch.randn(20000, 4, 8), torch.ones(20000, 4, dtype=torch.bool))
    null = stub_null(20000, 4)
    out = drop_prompts(prompt, null)
    dropped = (out.tokens == null.tokens).flatten(1).all(1)
    assert abs(dropped.float().mean() - PROMPT_DROPOUT) < 0.01
    assert torch.equal(dropped, (out.mask == null.mask).all(1))  # a row is dropped whole


def test_shift_moves_t_toward_noise():
    torch.manual_seed(0)
    assert abs(sample_t(50000, 1.0).median() - 0.5) < 0.01
    # a shift s sends the median 0.5 to s / (1 + s)
    assert abs(sample_t(50000, 3.0).median() - 0.75) < 0.01


def test_the_unweighted_loss_is_the_mse_and_a_white_error_keeps_its_loss_under_any_weight():
    torch.manual_seed(0)
    x, eps, x_t = torch.randn(3, 2, 64, 64), torch.randn(3, 2, 64, 64), torch.randn(3, 2, 64, 64)
    error = torch.randn(3, 2, 64, 64)
    out = target(x, eps, x_t) + error
    assert torch.allclose(per_sample_loss(out, x, eps, x_t), error.square().flatten(1).mean(1), rtol=1e-5)
    assert torch.allclose(per_sample_loss(out, x, eps, x_t, alpha=1.0), error.square().flatten(1).mean(1), rtol=0.1)


def test_flow_loss_trains_on_the_weighted_loss_and_reports_the_plain_one():
    x, prompt = torch.randn(8, 2, 32, 32), Prompt(torch.randn(8, 3, 8), torch.ones(8, 3, dtype=torch.bool))
    reads = []
    for alpha in (0.0, 1.0):
        torch.manual_seed(0)
        reads.append(flow_loss(ZeroNet(), x, prompt, 1.0, alpha=alpha))
    (plain, plain_per_sample, _), (weighted, weighted_per_sample, _) = reads
    torch.testing.assert_close(weighted_per_sample, plain_per_sample)
    torch.testing.assert_close(plain, plain_per_sample.mean())
    assert (weighted - plain).abs() > 0.01 * plain


def test_the_weight_charges_a_layout_error_more_than_a_detail_error_of_the_same_size():
    """At alpha 1 a wave of period 32 latent pixels costs more than one of period 4 by about (32 / 4)^β."""
    rows = torch.arange(64.0)[:, None].expand(64, 64)
    coarse, fine = [torch.cos(2 * torch.pi * rows / period).expand(1, 2, 64, 64) for period in (32, 4)]
    zero = torch.zeros(1, 2, 64, 64)
    cost = [per_sample_loss(e, zero, zero, zero, alpha=1.0).item() for e in (coarse, fine)]
    assert cost[0] / cost[1] == pytest.approx(8**1.52, rel=1e-3)
    assert spectral_weight(64, 64, 0.0).eq(1).all() and spectral_weight(64, 64, 1.0).mean().item() == pytest.approx(1)


def test_t_quantiles_are_the_training_t_at_equal_probability_steps():
    torch.manual_seed(0)
    for shift in (1.0, 3.0):
        p = (torch.arange(10) + 0.5) / 10
        assert (t_quantiles(10, shift) - torch.quantile(sample_t(200000, shift), p)).abs().max() < 0.005


def test_bucketed_loss():
    per_sample = torch.tensor([1.0, 3.0, 5.0, 7.0])
    t = torch.tensor([0.05, 0.15, 0.95, 1.0])
    buckets = loss_by_t_bucket(per_sample, t, buckets=10)
    assert buckets[0] == 1.0 and buckets[1] == 3.0 and buckets[9] == 6.0
    assert buckets[2:9].isnan().all()


class ZeroNet(torch.nn.Module):
    """F = 0, as at init; remembers the prompt it was given."""

    null_prompt = staticmethod(stub_null)

    def forward(self, x_t, t, prompt):
        self.prompt = prompt
        return torch.zeros_like(x_t)


def test_flow_loss_at_init_is_the_expected_target_energy_and_dropped_rows_get_the_null():
    # F = 0, so per element E[T²] = (2 - t)² + (1 - t)² for unit-normal x and ε.
    torch.manual_seed(0)
    net, prompt = ZeroNet(), Prompt(torch.randn(4000, 3, 8), torch.ones(4000, 3, dtype=torch.bool))
    loss, per_sample, t = flow_loss(net, torch.randn(4000, 2, 16, 16), prompt, shift=1.0)
    expected = ((2 - t) ** 2 + (1 - t) ** 2).mean()
    assert torch.allclose(loss, expected, rtol=0.01)
    assert torch.allclose(per_sample.mean(), loss)
    dropped = (net.prompt.tokens == -1).flatten(1).all(1)
    assert abs(dropped.float().mean() - PROMPT_DROPOUT) < 0.02
    assert all(torch.equal(a[~dropped], b[~dropped]) for a, b in zip(net.prompt, prompt))


@needs_cuda
def test_flow_loss_passes_gradient_to_the_network(model, prompt):
    model, prompt = model.cuda(), prompt.to("cuda", torch.float32)
    open_gates(model)  # closed, F = x_t and the meta channels the patch embed fills are not read
    loss, per_sample, t = flow_loss(model, torch.randn(3, 2, 64, 64, device="cuda"), prompt, shift=1.0)
    assert per_sample.shape == t.shape == (3,)
    loss.backward()
    assert model.patch_embed.weight.grad.abs().max() > 0


def test_sigma_narrows_the_logit_normal_around_its_shifted_median():
    wide, narrow = sample_t(50000, 1.5), sample_t(50000, 1.5, sigma=0.5)
    assert abs(narrow.median() - wide.median()) < 0.01 and abs(wide.median() - 0.6) < 0.01
    assert narrow.std() < 0.6 * wide.std()


def test_per_sample_loss_is_the_clean_latent_error_over_t_squared():
    torch.manual_seed(0)
    x, eps, x_hat, t = torch.randn(6, 2, 8, 8), torch.randn(6, 2, 8, 8), torch.randn(6, 2, 8, 8), torch.rand(6) * 0.9 + 0.05
    x_t = noisy(x, eps, t)
    out = x_t - (x_t - x_hat) / t.view(-1, 1, 1, 1)  # the F output of a model whose clean-latent prediction is x_hat
    torch.testing.assert_close(per_sample_loss(out, x, eps, x_t), (x - x_hat).square().flatten(1).mean(1) / t.square(), rtol=1e-4, atol=1e-5)


def test_explained_variance_is_zero_for_predicting_zero_and_one_for_exact():
    torch.manual_seed(0)
    x, eps, t = torch.randn(4000, 2, 8, 8), torch.randn(4000, 2, 8, 8), torch.rand(4000) * 0.9 + 0.05
    energy = x.square().flatten(1).mean(1)
    x_t = noisy(x, eps, t)
    zero_out = x_t - (x_t - torch.zeros_like(x)) / t.view(-1, 1, 1, 1)  # predicts x_hat = 0
    exact_out = x_t - (eps - x)
    assert abs(explained_variance(per_sample_loss(zero_out, x, eps, x_t), t, energy)) < 1e-4
    assert explained_variance(per_sample_loss(exact_out, x, eps, x_t), t, energy) > 1 - 1e-5


def test_explained_variance_by_bucket_reads_each_bin_and_puts_t_one_in_the_last():
    t, energy = torch.tensor([0.1, 0.3, 0.3, 1.0]), torch.tensor([1.0, 1.0, 2.0, 4.0])
    per_sample = torch.tensor([4.0, 1.0, 1.0, 0.25])  # clean-latent errors 0.04, 0.09, 0.09, 0.25
    by_bucket = explained_variance_by_t_bucket(per_sample, t, energy, buckets=2)
    assert by_bucket[0] == pytest.approx(1 - (0.04 + 0.09 + 0.09) / 4) and by_bucket[1] == pytest.approx(1 - 0.25 / 4)
    assert torch.isnan(explained_variance_by_t_bucket(per_sample[:3], t[:3], energy[:3], buckets=2)[1])


def test_flow_loss_samples_t_with_the_given_sigma():
    x, prompt = torch.randn(20000, 2, 8, 8), stub_null(20000, 2)
    wide, narrow = flow_loss(ZeroNet(), x, prompt, 1.0)[2], flow_loss(ZeroNet(), x, prompt, 1.0, sigma=0.5)[2]
    assert narrow.std() < 0.6 * wide.std()


@pytest.mark.parametrize("period, band", [(4, 2), (8, 1), (16, 1), (32, 0)])  # latent px; a token is 8
def test_a_wave_lands_in_the_band_of_its_period_in_tokens(period, band):
    wave = torch.cos(2 * math.pi * torch.arange(64) / period).expand(1, 2, 64, 64)
    energy = band_energy(wave)[0]
    torch.testing.assert_close(energy.sum(), wave.square().mean())  # the bands split the mean square
    assert energy[band] > 0.999 * energy.sum()
