import pytest
import torch
from conftest import open_gates

from tinydit.model import Prompt
from tinydit.sampler import euler_sample


def stub_prompt(batch, value, length=2):
    """A prompt for networks that read the pooled embedding only: `value` everywhere."""
    return Prompt(torch.full((batch, 768), float(value)), torch.full((batch, length, 8), float(value)), torch.ones(batch, length, dtype=torch.bool))


def stub_null(batch, length):
    return stub_prompt(batch, 0, length)


class Oracle:
    """Exact network for one fixed (x, ε): F = x_t - (ε - x), so v̂ = ε - x at every t."""

    def __init__(self, x, eps):
        self.x, self.eps = x, eps

    null_prompt = staticmethod(stub_null)

    def __call__(self, x_t, t, prompt):
        return x_t - (self.eps - self.x).repeat(2, 1, 1, 1)  # the sampler batches [prompt, null]


def test_oracle_velocity_recovers_x():
    torch.manual_seed(0)
    x, eps = torch.randn(2, 2, 16, 16), torch.randn(2, 2, 16, 16)
    out = euler_sample(Oracle(x, eps), eps, stub_prompt(2, 0), steps=7, guidance=4.0)
    assert torch.allclose(out, x, atol=1e-5)


def test_guidance_combines_conditional_and_null_predictions(model, prompt):
    open_gates(model)
    noise = torch.randn(3, 2, 64, 64)
    plain = euler_sample(model, noise, prompt, steps=2, guidance=1.0)
    guided = euler_sample(model, noise, prompt, steps=2, guidance=3.0)
    assert not torch.allclose(plain, guided)


def test_sampler_runs_on_the_bf16_network_in_fp32(model, prompt):
    open_gates(model)
    noise = torch.randn(3, 2, 64, 64)
    out = euler_sample(model.bfloat16(), noise, prompt, steps=2, guidance=3.0)
    assert out.dtype == torch.float32 and out.shape == noise.shape
    assert out.isfinite().all() and not torch.equal(out, noise)


class Probe:
    """text rows are ones and the null is zeros: the net returns 2·[text is ones] + t, a different value
    for text and null that depends on t, and records the t it was called with."""

    null_prompt = staticmethod(stub_null)

    def __init__(self):
        self.t_seen = []

    def __call__(self, x_t, t, prompt):
        self.t_seen.append(t.unique().tolist())
        return (2 * prompt.pooled[:, :1] + t[:, None]).view(-1, 1, 1, 1).expand_as(x_t)


@pytest.mark.parametrize("guidance", [0.0, 1.0, 3.0])
def test_guidance_is_null_plus_scale_times_text_minus_null(guidance):
    # One step from t=1 lands on the guided prediction: F_null = 1, F_text = 3.
    out = euler_sample(Probe(), torch.zeros(2, 2, 8, 8), stub_prompt(2, 1), steps=1, guidance=guidance)
    assert torch.allclose(out, torch.full_like(out, 1 + 2 * guidance))


def test_the_network_is_called_at_the_start_of_each_step_from_1_to_0():
    probe = Probe()
    euler_sample(probe, torch.zeros(2, 2, 8, 8), stub_prompt(2, 1), steps=4, guidance=1.0)
    assert probe.t_seen == [[1.0], [0.75], [0.5], [0.25]]
