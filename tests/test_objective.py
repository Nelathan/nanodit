import pytest
import torch
from conftest import open_gates

from nanodit.model import Prompt
from nanodit.objective import PROMPT_DROPOUT, drop_prompts, explained_variance, explained_variance_by_t_bucket, flow_loss, loss_by_t_bucket, noisy, per_sample_loss, sample_t, t_quantiles, target


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
    """A null prompt for networks without tokens: pooled ones, tokens of -1, only the first token real."""
    return Prompt(torch.ones(batch, 768), -torch.ones(batch, length, 8), (torch.arange(length) < 1).expand(batch, -1))


def test_prompt_dropout_rate():
    torch.manual_seed(0)
    prompt = Prompt(torch.randn(20000, 768), torch.randn(20000, 4, 8), torch.ones(20000, 4, dtype=torch.bool))
    null = stub_null(20000, 4)
    out = drop_prompts(prompt, null)
    dropped = (out.pooled == null.pooled).all(dim=-1)
    assert abs(dropped.float().mean() - PROMPT_DROPOUT) < 0.01
    assert torch.equal(dropped, (out.tokens == null.tokens).flatten(1).all(1)) and torch.equal(dropped, (out.mask == null.mask).all(1))  # a row is dropped whole


def test_shift_moves_t_toward_noise():
    torch.manual_seed(0)
    assert abs(sample_t(50000, 1.0).median() - 0.5) < 0.01
    # a shift s sends the median 0.5 to s / (1 + s)
    assert abs(sample_t(50000, 3.0).median() - 0.75) < 0.01


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
    net, prompt = ZeroNet(), Prompt(torch.randn(4000, 768), torch.randn(4000, 3, 8), torch.ones(4000, 3, dtype=torch.bool))
    loss, per_sample, t = flow_loss(net, torch.randn(4000, 2, 16, 16), prompt, shift=1.0)
    expected = ((2 - t) ** 2 + (1 - t) ** 2).mean()
    assert torch.allclose(loss, expected, rtol=0.01)
    assert torch.allclose(per_sample.mean(), loss)
    dropped = (net.prompt.pooled == 1).all(dim=-1)
    assert abs(dropped.float().mean() - PROMPT_DROPOUT) < 0.02
    assert all(torch.equal(a[~dropped], b[~dropped]) for a, b in zip(net.prompt, prompt))


def test_flow_loss_passes_gradient_to_the_network(model, prompt):
    open_gates(model)  # closed, F = x_t and the meta channels the patch embed fills are not read
    loss, per_sample, t = flow_loss(model, torch.randn(3, 2, 64, 64), prompt, shift=1.0)
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
