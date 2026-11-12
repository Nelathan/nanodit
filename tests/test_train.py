import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from conftest import CROSS_PAIR, open_gates
from test_data import shuffle_module, write_shard
from nanodit.data import pad_sequences
from nanodit.model import DiT, NULL_TOKENS, SEQ_DIM, Prompt
from nanodit.objective import flow_loss as real_flow_loss
from nanodit.objective import noisy, t_quantiles, target
from nanodit.optim import build_optimizers

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_prompt(n, generator=None):
    tokens, mask = pad_sequences([torch.randn(2 + i % 4, SEQ_DIM, generator=generator) for i in range(n)], torch.float32)
    return Prompt(torch.nn.functional.normalize(torch.randn(n, 768, generator=generator), dim=-1), tokens, mask)


def write_config(tmp_path, steps, eval_every, checkpoint_every=100, batch=4, micro_batch=4, log_every=1):
    torch.manual_seed(0)
    for name, n in (("s", 16), ("eval", 5)):  # the training set and the eval set are both shuffled files
        write_shard(tmp_path / f"{name}_shard.parquet", torch.randn(n, 2, 16, 16).bfloat16(), torch.randn(n, 768).bfloat16(), [torch.randn(2 + i % 4, SEQ_DIM).bfloat16() for i in range(n)])
        shuffle_module(buckets=1).shuffle([str(tmp_path / f"{name}_shard.parquet")], 32, tmp_path / f"{name}.parquet")
    save_file({"null": torch.randn(768), "null_tokens": torch.randn(NULL_TOKENS, SEQ_DIM)}, tmp_path / "null.safetensors")
    (tmp_path / "prompts.txt").write_text("a red apple\na blue car\na cat\n")
    config = tmp_path / "config.toml"  # eval_prompts resolves next to it
    config.write_text(f"""
name = "smoke"
runs_dir = "{tmp_path / 'runs'}"
resolution = 32
dataset = "{tmp_path / 's.parquet'}"
null_embedding = "{tmp_path / 'null.safetensors'}"
depth = 2
width = 512
layout = ["conv", "attn"]
block_types = {{ conv = {{ mixer = "conv" }}, attn = {{ mixer = "attn", cross = true }} }}
batch = {batch}
micro_batch = {micro_batch}
steps = {steps}
seed = 0
usui_lr = 3e-3
beta = 0.95
rank_fraction = 0.1
fallback_lr = 1e-3
warmup = 2
shift = 1.5
log_every = {log_every}
checkpoint_every = {checkpoint_every}
eval_every = {eval_every}
eval_prompts = "prompts.txt"
eval_seed = 42
guidance = 2.0
sampler_steps = 2
eval_set = "{tmp_path / 'eval.parquet'}"
wandb_project = "test"
""")
    return config


class StubDecoder:
    """Stands in for Primal: [B, 2, h, w] latent -> [B, 3, 2h, 2w] image."""

    def decode(self, z):
        return torch.nn.functional.interpolate(torch.cat([z, z[:, :1]], 1), scale_factor=2)


def stub_models(monkeypatch, train):
    monkeypatch.setenv("WANDB_MODE", "disabled")  # tests never reach the network
    monkeypatch.setattr(train, "embed_prompts", lambda prompts: random_prompt(len(prompts)))
    monkeypatch.setattr(train, "load_decoder", lambda device: StubDecoder())


def test_train_writes_logs_eval_grid_and_loadable_checkpoint(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    sent = []
    # wandb.init rebinds wandb.log to its run's, so the run is stubbed whole
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(train.wandb, "log", lambda data, step: sent.append((step, data)))
    monkeypatch.setattr(train.wandb, "finish", lambda: None)
    step_losses = []

    def accumulate_gradients(*args):
        loss, per_sample, t, energy = real_accumulate(*args)
        step_losses.append(loss.item())
        return loss, per_sample, t, energy

    real_accumulate = train.accumulate_gradients
    monkeypatch.setattr(train, "accumulate_gradients", accumulate_gradients)
    noises = []
    real_sample = train.euler_sample
    monkeypatch.setattr(train, "euler_sample", lambda model, noise, *rest: noises.append(noise.clone()) or real_sample(model, noise, *rest))
    train.main(str(write_config(tmp_path, steps=3, eval_every=2, checkpoint_every=2, log_every=2)))
    run = tmp_path / "runs" / "smoke"

    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [2, 3]  # every log_every, and the last step
    # a record averages the steps since the last one
    assert abs(log[0]["loss"] - sum(step_losses[:2]) / 2) < 1e-6 and abs(log[1]["loss"] - step_losses[2]) < 1e-6
    assert sum(v is not None for v in log[0]["loss_by_t"]) <= 8  # 4 samples per step, two steps
    assert all(len(r["loss_by_t"]) == 10 and r["fallback_grad_norm"] > 0 and r["imgs_per_s"] > 0 for r in log)
    # under Molegrad the raw gradient norm is a probed estimate, so capture can pass 1 (FACTS, GPU smoke runs);
    # random frames: a matrix behind a still-closed zero gate reads capture 0, which pulls the mean down in a 3-step run
    assert all(0 < r["usui"]["grad_capture"] < 1.5 for r in log)
    assert "tangent_live_fraction" in log[1]["usui"]
    assert (run / "config.toml").exists() and (run / "eval_prompts.txt").read_text() == "a red apple\na blue car\na cat\n"
    assert [r["lr_usui"] for r in log] == [3e-3, 3e-3] and [r["lr_fallback"] for r in log] == [1e-3, 1e-3]  # the last step's, past warmup 2

    curves = [(step, data) for step, data in sent if "train/loss" in data]
    assert [step for step, _ in curves] == [2, 3] and [data["train/loss"] for _, data in curves] == [r["loss"] for r in log]
    for (_, data), r in zip(curves, log):  # the same values as train.jsonl, empty t buckets left out
        expected = {f"train/{key}": r[key] for key in ("loss", "r2", "lr_usui", "lr_fallback", "fallback_grad_norm", "imgs_per_s")}
        by_t = {key: value for key, value in data.items() if "_by_t/" in key}
        edges = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
        assert all(key.split("/")[0] in ("train_loss_by_t", "train_r2_by_t") and key.split("/")[1] in [f"t{e}" for e in edges] for key in by_t)
        for key, value in by_t.items():  # a wandb bucket pools its pair of train.jsonl buckets: within their range
            group, bucket = key.split("/")
            first = edges.index(bucket[1:]) * 2
            pair = [v for v in r[group.removeprefix("train_")][first : first + 2] if v is not None]
            assert min(pair) - 1e-6 <= value <= max(pair) + 1e-6
        assert {key: value for key, value in data.items() if "_by_t/" not in key} == expected | {f"usuitrack/{key}": v for key, v in r["usui"].items()}
    assert [step for step, data in sent if "eval" in data] == [2, 3]

    scores = [json.loads(line) for line in (run / "score.jsonl").read_text().splitlines()]
    assert [s["step"] for s in scores] == [2, 3]  # every eval_every, and the last step: not only the checkpoints
    assert all(s["t"] == pytest.approx(t_quantiles(10, 1.0).tolist()) for s in scores)  # the shift-1 grid, though the run trains at shift 1.5
    logged = [(step, data) for step, data in sent if "eval/loss_caption" in data]
    assert [step for step, _ in logged] == [2, 3]
    for (_, data), s in zip(logged, scores):  # the means, and each by-t series in fifths
        assert data["eval/r2_caption"] == s["r2_caption"] and data["eval/loss_caption"] == s["loss_caption"]
        assert data["eval_r2_caption_by_t/q0.2-0.4"] == pytest.approx((s["r2_caption_by_t"][2] + s["r2_caption_by_t"][3]) / 2)
        weights = {"weights/norm": s["weights"]["norm"], "weights/norm_ratio": s["weights"]["norm_ratio"]} | {f"stable_rank/{role}": v for role, v in s["weights"]["stable_rank"].items()}
        assert {key: value for key, value in data.items() if key in weights} == weights
        assert len(data) == 2 + 2 * 5 + len(weights)  # the two means, the loss and R² in fifths, the weight reads
    for s in scores:  # the ceiling is the init norm: the ratio is at most 1, up to bf16 rounding
        assert 0 < s["weights"]["norm_ratio"] < 1 + 2e-3
        assert {"conv.out", "attn.out", "attn.cross_kv", "ada.1", "patch_embed"} <= set(s["weights"]["stable_rank"])
        assert all(0 < v <= 1 for v in s["weights"]["stable_rank"].values())
    for s in scores:
        assert len(s["t"]) == len(s["loss_caption_by_t"]) == len(s["r2_caption_by_t"]) == 10 and "loss_null" not in s

    evals = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [2, 3]  # every eval_every, and the last step
    assert evals[0]["seeds"] == [42, 43, 44]
    # each prompt its own noise, the same at every eval
    assert torch.equal(noises[0], noises[1]) and not torch.equal(noises[0][0], noises[0][1])
    from PIL import Image
    grid = Image.open(evals[0]["image"])
    assert grid.size == (2 * 32, 2 * 32)  # three prompts, two columns, row-major
    assert grid.crop((32, 32, 64, 64)).getextrema() == ((0, 0),) * 3  # the cell past the last prompt stays black

    checkpoint = run / "checkpoints" / "step_0000003"
    first = load_file(run / "checkpoints" / "step_0000002" / "model.safetensors")
    last = load_file(checkpoint / "model.safetensors")
    assert not torch.equal(first["blocks.1.qkv.weight"], last["blocks.1.qkv.weight"])  # a UsuiTrack matrix moved

    assert {t.dtype for t in last.values()} == {torch.bfloat16}  # the checkpoint holds the bf16 state
    restored = DiT(torch.zeros(768), torch.zeros(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=CROSS_PAIR).bfloat16()
    restored.load_state_dict(last)
    assert restored.blocks[0].table[[2, 5]].abs().max() > 0  # trained, not the zero init
    usui_state, adam_state = torch.load(checkpoint / "optimizer.pt")
    matrix, fallback = build_optimizers(restored, 3e-3, 0.95, 0.1, 1e-3)
    matrix.load_state_dict(usui_state)
    fallback.load_state_dict(adam_state)


def test_score_is_the_caption_loss_at_each_t_with_each_rows_own_noise(train):
    class Echo(torch.nn.Module):  # its output depends on t and on the embedding it is given
        def forward(self, x_t, t, prompt):
            return x_t * t.view(-1, 1, 1, 1) + prompt.pooled[:, :1, None, None]

    torch.manual_seed(0)
    x, noise, ts = torch.randn(5, 2, 4, 4).bfloat16(), torch.randn(5, 2, 4, 4), torch.tensor([0.2, 0.7])
    prompt = random_prompt(5)
    result = train.score(Echo(), x, prompt, noise, ts, 2, torch.device("cpu"))
    by_t = []
    for t_value in ts.tolist():
        t = torch.full((5,), t_value)
        x_t = noisy(x.float(), noise, t)
        by_t.append(float((Echo()(x_t, t, prompt) - target(x.float(), noise, x_t)).square().mean()))
    torch.testing.assert_close(result["loss_caption_by_t"], by_t)
    torch.testing.assert_close(result["loss_caption"], sum(by_t) / 2)
    energy = float(x.float().square().mean())  # R²: one minus the clean-latent error, loss · t², over the latents' energy
    r2_by_t = [1 - loss * t_value**2 / energy for loss, t_value in zip(by_t, ts.tolist())]
    torch.testing.assert_close(result["r2_caption_by_t"], r2_by_t)
    torch.testing.assert_close(result["r2_caption"], sum(r2_by_t) / 2)
    torch.testing.assert_close(train.score(Echo(), x, prompt, noise, ts, 5, torch.device("cpu")), result)  # a partial last batch changes nothing


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


def test_non_finite_loss_stops_the_run_at_the_next_record_and_names_its_step(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=3, eval_every=3, checkpoint_every=3, log_every=3)
    calls = []

    def flow_loss(*args):
        loss, per_sample, t = real_flow_loss(*args)
        calls.append(None)
        return (loss * float("nan") if len(calls) == 2 else loss), per_sample, t

    monkeypatch.setattr(train, "flow_loss", flow_loss)
    with pytest.raises(RuntimeError, match="non-finite .* at step 2"):
        train.main(str(config))
    assert not (tmp_path / "runs" / "smoke" / "train.jsonl").exists()


def test_accumulated_batch_trains_logs_and_checkpoints(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    train.main(str(write_config(tmp_path, steps=3, eval_every=5, checkpoint_every=3, batch=8, micro_batch=2)))
    run = tmp_path / "runs" / "smoke"
    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3]
    assert all(len(r["loss_by_t"]) == 10 and r["loss"] > 0 for r in log)
    assert [r["lr_usui"] for r in log] == [1.5e-3, 3e-3, 3e-3] and [r["lr_fallback"] for r in log] == [5e-4, 1e-3, 1e-3]  # warmup 2
    assert 0 < sum(v is not None for r in log for v in r["loss_by_t"]) <= 24  # 8 samples per step, 4 micro-batches
    assert (run / "checkpoints" / "step_0000003" / "optimizer.pt").exists()


def test_a_run_from_a_checkpoint_starts_from_its_weights_and_optimizer_state_at_its_own_lr(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=2, eval_every=2, checkpoint_every=2)
    train.main(str(config))
    saved = tmp_path / "runs" / "smoke" / "checkpoints" / "step_0000002"
    text = config.read_text().replace('name = "smoke"', 'name = "resumed"').replace("usui_lr = 3e-3", "usui_lr = 1e-3")
    config.write_text(text.replace("steps = 2", "steps = 1").replace("warmup = 2", "warmup = 0") + f'init_from = "{saved}"\n')
    seen = []
    real_accumulate = train.accumulate_gradients

    def accumulate_gradients(model, stream, micro_batches, *rest):
        if not seen:  # before the first step
            seen.append(({k: v.clone() for k, v in model.state_dict().items()}, [copy.deepcopy(o.state_dict()) for o in optimizers]))  # state_dict holds the live tensors
        return real_accumulate(model, stream, micro_batches, *rest)

    optimizers = []
    real_build = train.build_optimizers
    monkeypatch.setattr(train, "build_optimizers", lambda *args: optimizers.extend(real_build(*args)) or tuple(optimizers))
    monkeypatch.setattr(train, "accumulate_gradients", accumulate_gradients)
    train.main(str(config))

    weights, states = seen[0]
    assert all(torch.equal(weights[k].cpu(), v) for k, v in load_file(saved / "model.safetensors").items())
    for state, saved_state in zip(states, torch.load(saved / "optimizer.pt"), strict=True):
        assert state["state"].keys() == saved_state["state"].keys()
        assert all(state["state"][i]["step"] == s["step"] for i, s in saved_state["state"].items() if "step" in s)  # AdamW's bias correction continues
        assert all(torch.equal(state["state"][i][key].cpu(), value.cpu()) for i, s in saved_state["state"].items() for key, value in s.items() if torch.is_tensor(value))
    run = tmp_path / "runs" / "resumed"
    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [3] and log[0]["lr_usui"] == 1e-3 and log[0]["lr_fallback"] == 1e-3
    assert (run / "checkpoints" / "step_0000003").exists() and [json.loads(line)["step"] for line in (run / "score.jsonl").read_text().splitlines()] == [3]


def test_eval_and_checkpoint_fall_on_a_record(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=4, eval_every=3, checkpoint_every=4, log_every=2)
    with pytest.raises(ValueError, match="log_every"):
        train.main(str(config))
    assert not (tmp_path / "runs" / "smoke").exists()


def test_batch_must_be_a_multiple_of_micro_batch(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=1, eval_every=5, batch=6, micro_batch=4)
    (tmp_path / "s.parquet").unlink()  # checked before the dataset is read
    with pytest.raises(ValueError, match="multiple"):
        train.main(str(config))
    assert not (tmp_path / "runs" / "smoke").exists()


class FixedBatches:
    """Hands out the given micro-batches in order, like the Stream does for one step."""

    def __init__(self, chunks):
        self.chunks, self.i = chunks, 0

    def next(self):
        self.i += 1
        return self.chunks[(self.i - 1) % len(self.chunks)]


def fixed_flow_loss(model, x, prompt, shift, sigma):
    """flow_loss with t and ε a function of the sample, so any split of a batch sees the same draws."""
    t = torch.sigmoid(x.mean((1, 2, 3)) * 10)
    eps = x.flip(-1)
    x_t = noisy(x, eps, t)
    per_sample = (model(x_t, t, prompt) - target(x, eps, x_t)).square().flatten(1).mean(1)
    return per_sample.mean(), per_sample, t


def test_accumulation_covers_the_whole_batch_and_leaves_the_mean_gradient(train, monkeypatch):
    monkeypatch.setattr(train, "flow_loss", fixed_flow_loss)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 2, 16, 16, generator=g)
    prompt = random_prompt(8, g)
    results = []
    for micro_batches in (1, 4):  # the same 8 samples as one batch, and as 4 micro-batches of 2
        torch.manual_seed(1)
        model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=CROSS_PAIR)
        open_gates(model)
        build_optimizers(model.bfloat16(), 3e-3, 0.95, 0.1, 1e-3)
        size = 8 // micro_batches
        chunks = [(x[i : i + size], prompt.rows(i, i + size)) for i in range(0, 8, size)]
        loss, per_sample, t, _ = train.accumulate_gradients(model, FixedBatches(chunks), micro_batches, shift=1.0, sigma=1.0)
        results.append((loss, per_sample, t, [model.patch_embed.bias.grad, model.blocks[0].table.grad]))
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
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=CROSS_PAIR)
    open_gates(model)
    model.bfloat16()
    optimizers = build_optimizers(model, 3e-3, 0.95, 0.1, 1e-3)
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 2, 16, 16, generator=g)
    prompt = random_prompt(8, g)
    for _ in range(3):
        loss, per_sample, t, _ = train.accumulate_gradients(model, FixedBatches([(x[:4], prompt.rows(0, 4)), (x[4:], prompt.rows(4, 8))]), 2, shift=1.0, sigma=1.0)
        assert loss.dtype == per_sample.dtype == torch.float32 and loss.isfinite() and per_sample.isfinite().all()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    # The QK-norm gain (1.0, ulp 2^-7) takes AdamW steps of 1e-3, a fraction of an ulp: round-to-nearest
    # leaves it bit-identical (checked: 0 of its elements move), stochastic rounding moves about a quarter.
    changed = {n: (model.get_parameter(n) != start[n]).float().mean() for n in ("blocks.1.q_norm.weight", "blocks.1.qkv.weight", "blocks.1.w_down.weight", "patch_embed.weight")}
    assert changed["blocks.1.q_norm.weight"] > 0.1
    assert all(fraction > 0.02 for fraction in changed.values()), changed


def test_a_resumed_run_takes_its_lr_from_the_config_not_the_checkpoint(train, tmp_path):
    torch.manual_seed(0)
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=CROSS_PAIR).bfloat16()
    open_gates(model)
    fresh = copy.deepcopy(model)  # Molegrad attaches to a model once
    saved = build_optimizers(model, 1e-3, 0.95, 0.1, 1e-3)
    train.save_checkpoint(model, saved, 7, tmp_path)
    model = fresh
    resumed = build_optimizers(model, 2e-3, 0.95, 0.1, 1e-3)
    assert train.load_checkpoint(model, resumed, tmp_path / "checkpoints" / "step_0000007") == 7
    assert {g["lr"] for g in resumed[0].param_groups} == {2e-3}


def test_scores_follow_eval_every_and_checkpoints_follow_checkpoint_every(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(train.wandb, "log", lambda data, step: None)
    monkeypatch.setattr(train.wandb, "finish", lambda: None)
    train.main(str(write_config(tmp_path, steps=3, eval_every=1, checkpoint_every=100)))
    run = tmp_path / "runs" / "smoke"
    assert [json.loads(line)["step"] for line in (run / "score.jsonl").read_text().splitlines()] == [1, 2, 3]
    assert [p.name for p in (run / "checkpoints").iterdir()] == ["step_0000003"]


def test_wandb_buckets_pool_the_interval_rather_than_averaging_the_train_buckets(train):
    interval = train.Interval()
    t = torch.tensor([0.05, 0.15, 0.15, 0.15, 0.15, 0.15, 0.9, 1.0])  # 1 sample in [0, 0.1), 5 in [0.1, 0.2), 2 at the top
    per_sample, energy = torch.tensor([10.0, 1, 1, 1, 1, 1, 2, 4]), torch.ones(8)
    interval.add(per_sample.mean(), per_sample, t, energy, torch.tensor(0.0))
    by_t = interval.by_t(5)
    assert by_t["loss_by_t"][0] == pytest.approx(15 / 6) and by_t["loss_by_t"][1:4] == [None] * 3 and by_t["loss_by_t"][4] == pytest.approx(3.0)  # t = 1.0 lands in the last bucket
    error = per_sample * t.square()
    assert by_t["r2_by_t"][0] == pytest.approx(1 - error[:6].sum().item() / 6) and by_t["r2_by_t"][4] == pytest.approx(1 - error[6:].sum().item() / 2)


def test_the_lr_is_constant_after_warmup(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=4, eval_every=100)
    train.main(str(config))
    log = [json.loads(line) for line in (tmp_path / "runs" / "smoke" / "train.jsonl").read_text().splitlines()]
    assert [r["lr_usui"] for r in log] == pytest.approx([1.5e-3, 3e-3, 3e-3, 3e-3])  # warmup 2


def test_an_anneal_takes_both_lrs_down_a_cosine_to_the_usui_target(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    config = write_config(tmp_path, steps=4, eval_every=100)
    config.write_text(config.read_text().replace("warmup = 2", "warmup = 0\nanneal_to = 1e-5"))
    train.main(str(config))
    log = [json.loads(line) for line in (tmp_path / "runs" / "smoke" / "train.jsonl").read_text().splitlines()]
    end = 1e-5 / 3e-3
    factors = [end + (1 - end) * (1 + math.cos(math.pi * done / 4)) / 2 for done in range(4)]
    assert [r["lr_usui"] for r in log] == pytest.approx([3e-3 * f for f in factors])
    assert [r["lr_fallback"] for r in log] == pytest.approx([1e-3 * f for f in factors])
