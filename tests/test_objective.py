import torch

from tinydit.objective import PROMPT_DROPOUT, drop_prompts, flow_loss, loss_by_t_bucket, noisy, sample_t, t_quantiles, target


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


def test_prompt_dropout_rate():
    torch.manual_seed(0)
    text, null = torch.randn(20000, 768), torch.randn(768)
    dropped = (drop_prompts(text, null) == null).all(dim=-1).float().mean()
    assert abs(dropped - PROMPT_DROPOUT) < 0.01


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
    """F = 0, as at init; remembers the text rows it was given."""

    def __init__(self):
        super().__init__()
        self.null_embedding = torch.nn.Parameter(torch.randn(768))

    def forward(self, x_t, t, text):
        self.text = text
        return torch.zeros_like(x_t)


def test_flow_loss_at_init_is_the_expected_target_energy_and_dropped_rows_get_the_null():
    # F = 0, so per element E[T²] = (2 - t)² + (1 - t)² for unit-normal x and ε.
    torch.manual_seed(0)
    net, text = ZeroNet(), torch.randn(4000, 768)
    loss, per_sample, t = flow_loss(net, torch.randn(4000, 2, 16, 16), text, shift=1.0)
    expected = ((2 - t) ** 2 + (1 - t) ** 2).mean()
    assert torch.allclose(loss, expected, rtol=0.01)
    assert torch.allclose(per_sample.mean(), loss)
    dropped = (net.text == net.null_embedding).all(dim=-1)
    assert abs(dropped.float().mean() - PROMPT_DROPOUT) < 0.02
    assert torch.equal(net.text[~dropped], text[~dropped])


def test_flow_loss_passes_gradient_to_the_network(model, text):
    loss, per_sample, t = flow_loss(model, torch.randn(3, 2, 64, 64), text, shift=1.0)
    assert per_sample.shape == t.shape == (3,)
    loss.backward()
    assert model.final.weight.grad.abs().max() > 0
