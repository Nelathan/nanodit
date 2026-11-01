import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_data import write_shard
from tinydit.model import DiT
from tinydit.objective import flow_loss as real_flow_loss
from tinydit.objective import noisy, target
from tinydit.optim import build_optimizers

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path, steps, eval_every, checkpoint_every=100, batch=4, micro_batch=4):
    torch.manual_seed(0)
    write_shard(tmp_path / "s.parquet", torch.randn(16, 2, 16, 16).bfloat16(), torch.randn(16, 768).bfloat16())
    save_file({"null": torch.randn(768)}, tmp_path / "null.safetensors")
    (tmp_path / "prompts.txt").write_text("a red apple\na blue car\n")
    config = tmp_path / "config.toml"  # eval_prompts resolves next to it
    config.write_text(f"""
name = "smoke"
runs_dir = "{tmp_path / 'runs'}"
resolution = 32
parquet_glob = "{tmp_path / '*.parquet'}"
null_embedding = "{tmp_path / 'null.safetensors'}"
depth = 2
batch = {batch}
micro_batch = {micro_batch}
steps = {steps}
seed = 0
usui_lr = 3e-3
beta = 0.95
rank_fraction = 0.1
fallback_lr = 1e-3
warmup = 2
shift = 1.0
checkpoint_every = {checkpoint_every}
eval_every = {eval_every}
eval_prompts = "prompts.txt"
eval_seeds = [0, 1, 2]
guidance = 2.0
sampler_steps = 2
wandb_project = "test"
""")
    return config


class StubDecoder:
    """Stands in for Primal: [B, 2, h, w] latent -> [B, 3, 2h, 2w] image."""

    def decode(self, z):
        return torch.nn.functional.interpolate(torch.cat([z, z[:, :1]], 1), scale_factor=2)


def stub_models(monkeypatch, train):
    monkeypatch.setenv("WANDB_MODE", "disabled")  # tests never reach the network
    monkeypatch.setattr(train, "embed_prompts", lambda prompts: torch.nn.functional.normalize(torch.randn(len(prompts), 768), dim=-1))
    monkeypatch.setattr(train, "load_decoder", lambda device: StubDecoder())


def test_train_writes_logs_eval_grid_and_loadable_checkpoint(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    sent = []
    # wandb.init rebinds wandb.log to its run's, so the run is stubbed whole
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(train.wandb, "log", lambda data, step: sent.append((step, data)))
    monkeypatch.setattr(train.wandb, "finish", lambda: None)
    train.main(str(write_config(tmp_path, steps=3, eval_every=2, checkpoint_every=1)))
    run = tmp_path / "runs" / "smoke"

    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3]
    assert all(len(r["loss_by_t"]) == 10 and r["loss"] > 0 and r["fallback_grad_norm"] > 0 and r["imgs_per_s"] > 0 for r in log)
    # under Molegrad the raw gradient norm is a probed estimate, so capture can pass 1 (FACTS, GPU smoke runs)
    assert all(0.5 < r["usui"]["grad_capture"] < 1.5 for r in log)
    assert "tangent_live_fraction" in log[2]["usui"]
    assert (run / "config.toml").exists() and (run / "eval_prompts.txt").read_text() == "a red apple\na blue car\n"
    assert [r["lr_usui"] for r in log] == [1.5e-3, 3e-3, 3e-3] and [r["lr_fallback"] for r in log] == [5e-4, 1e-3, 1e-3]  # warmup 2

    curves = [(step, data) for step, data in sent if "eval" not in data]
    assert [step for step, _ in curves] == [1, 2, 3] and [data["loss"] for _, data in curves] == [r["loss"] for r in log]
    for (_, data), r in zip(curves, log):  # the same values as train.jsonl, empty t buckets left out
        expected = {key: r[key] for key in ("loss", "lr_usui", "lr_fallback", "fallback_grad_norm", "imgs_per_s")}
        expected |= {f"loss_by_t/{b / 10:.1f}": v for b, v in enumerate(r["loss_by_t"]) if v is not None}
        assert data == expected | {f"usuitrack/{key}": v for key, v in r["usui"].items()}
    assert [step for step, data in sent if "eval" in data] == [2, 3]

    evals = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [2, 3]  # every eval_every, and the last step
    from PIL import Image
    assert Image.open(evals[0]["image"]).size == (3 * 32, 2 * 32)  # seeds across, prompts down

    checkpoint = run / "checkpoints" / "step_0000003"
    first = load_file(run / "checkpoints" / "step_0000001" / "model.safetensors")
    last = load_file(checkpoint / "model.safetensors")
    assert not torch.equal(first["blocks.0.qkv.weight"], last["blocks.0.qkv.weight"])  # a UsuiTrack matrix moved

    assert {t.dtype for t in last.values()} == {torch.bfloat16}  # the checkpoint holds the bf16 state
    restored = DiT(torch.zeros(768), depth=2).bfloat16()
    restored.load_state_dict(last)
    assert restored.final.weight.abs().max() > 0  # trained, not the zero init
    usui_state, adam_state = torch.load(checkpoint / "optimizer.pt")
    matrix, fallback = build_optimizers(restored, 3e-3, 0.95, 0.1, 1e-3)
    matrix.load_state_dict(usui_state)
    fallback.load_state_dict(adam_state)


def test_existing_run_is_not_overwritten(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=1, eval_every=5)
    train.main(str(config))
    with pytest.raises(FileExistsError):
        train.main(str(config))


def test_failed_start_leaves_no_run_directory(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=1, eval_every=5)
    (tmp_path / "null.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        train.main(str(config))
    assert not (tmp_path / "runs" / "smoke").exists()


def test_non_finite_loss_stops_before_any_optimizer_step(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=2, eval_every=5)
    monkeypatch.setattr(train, "flow_loss", lambda *a: (lambda l, p, t: (l * float("nan"), p, t))(*real_flow_loss(*a)))
    with pytest.raises(RuntimeError, match="non-finite"):
        train.main(str(config))
    assert not (tmp_path / "runs" / "smoke" / "train.jsonl").exists()


def test_accumulated_batch_trains_logs_and_checkpoints(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    train.main(str(write_config(tmp_path, steps=3, eval_every=5, checkpoint_every=3, batch=8, micro_batch=2)))
    run = tmp_path / "runs" / "smoke"
    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3]
    assert all(len(r["loss_by_t"]) == 10 and r["loss"] > 0 for r in log)
    assert 0 < sum(v is not None for r in log for v in r["loss_by_t"]) <= 24  # 8 samples per step, 4 micro-batches
    assert (run / "checkpoints" / "step_0000003" / "optimizer.pt").exists()


def test_batch_must_be_a_multiple_of_micro_batch(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=1, eval_every=5, batch=6, micro_batch=4)
    (tmp_path / "s.parquet").unlink()  # checked before the dataset is read
    with pytest.raises(ValueError, match="multiple"):
        train.main(str(config))
    assert not (tmp_path / "runs" / "smoke").exists()


class FixedBatches:
    """Hands out the given micro-batches in order, like the Batcher does for one step."""

    def __init__(self, chunks):
        self.chunks, self.i = chunks, 0

    def next(self):
        self.i += 1
        return self.chunks[(self.i - 1) % len(self.chunks)]


def fixed_flow_loss(model, x, text, shift):
    """flow_loss with t and ε a function of the sample, so any split of a batch sees the same draws."""
    t = torch.sigmoid(x.mean((1, 2, 3)) * 10)
    eps = x.flip(-1)
    x_t = noisy(x, eps, t)
    per_sample = (model(x_t, t, text) - target(x, eps, x_t)).square().flatten(1).mean(1)
    return per_sample.mean(), per_sample, t


def test_accumulation_covers_the_whole_batch_and_leaves_the_mean_gradient(train, monkeypatch):
    monkeypatch.setattr(train, "flow_loss", fixed_flow_loss)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 2, 16, 16, generator=g)
    text = torch.nn.functional.normalize(torch.randn(8, 768, generator=g), dim=-1)
    results = []
    for micro_batches in (1, 4):  # the same 8 samples as one batch, and as 4 micro-batches of 2
        torch.manual_seed(1)
        model = DiT(torch.randn(768), depth=2)
        torch.nn.init.normal_(model.final.weight, std=0.05)
        build_optimizers(model.bfloat16(), 3e-3, 0.95, 0.1, 1e-3)
        size = 8 // micro_batches
        chunks = [(x[i : i + size], text[i : i + size]) for i in range(0, 8, size)]
        loss, per_sample, t = train.accumulate_gradients(model, FixedBatches(chunks), micro_batches, shift=1.0)
        results.append((loss, per_sample, t, [model.final.bias.grad, model.final_table.grad]))
    (loss1, per1, t1, grads1), (loss4, per4, t4, grads4) = results
    assert per4.shape == t4.shape == (8,)
    # bf16 batches the same samples differently, so equal only up to rounding; a lost or doubled
    # micro-batch would be off by tens of percent
    assert torch.allclose(per1, per4, rtol=0.02) and torch.equal(t1, t4)
    assert torch.allclose(loss1, loss4, rtol=0.02)
    for full, accumulated in zip(grads1, grads4):
        assert full.abs().max() > 0 and (full.float() - accumulated.float()).norm() < 0.03 * full.float().norm()


def test_bf16_model_trains_with_finite_loss_stays_bf16_and_its_updates_land(train):
    torch.manual_seed(0)
    model = DiT(torch.randn(768), depth=2)
    torch.nn.init.normal_(model.final.weight, std=0.05)
    model.bfloat16()
    optimizers = build_optimizers(model, 3e-3, 0.95, 0.1, 1e-3)
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 2, 16, 16, generator=g)
    text = torch.nn.functional.normalize(torch.randn(8, 768, generator=g), dim=-1)
    for _ in range(3):
        loss, per_sample, t = train.accumulate_gradients(model, FixedBatches([(x[:4], text[:4]), (x[4:], text[4:])]), 2, shift=1.0)
        assert loss.dtype == per_sample.dtype == torch.float32 and loss.isfinite() and per_sample.isfinite().all()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    # The QK-norm gain (1.0, ulp 2^-7) takes AdamW steps of 1e-3, a fraction of an ulp: round-to-nearest
    # leaves it bit-identical (checked: 0 of its elements move), stochastic rounding moves about a quarter.
    changed = {n: (model.get_parameter(n) != start[n]).float().mean() for n in ("blocks.0.q_norm.weight", "blocks.0.qkv.weight", "blocks.1.w_down.weight", "final.weight")}
    assert changed["blocks.0.q_norm.weight"] > 0.1
    assert all(fraction > 0.02 for fraction in changed.values()), changed
