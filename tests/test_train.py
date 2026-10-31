import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_data import write_shard
from tinydit.model import DiT
from tinydit.objective import flow_loss as real_flow_loss
from tinydit.optim import build_optimizers

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path, steps, eval_every, checkpoint_every=100):
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
batch = 4
steps = {steps}
seed = 0
usui_lr = 3e-3
beta = 0.95
fallback_lr = 1e-3
shift = 1.0
checkpoint_every = {checkpoint_every}
eval_every = {eval_every}
eval_prompts = "prompts.txt"
eval_seeds = [0, 1, 2]
guidance = 2.0
sampler_steps = 2
""")
    return config


class StubDecoder:
    """Stands in for Primal: [B, 2, h, w] latent -> [B, 3, 2h, 2w] image."""

    def decode(self, z):
        return torch.nn.functional.interpolate(torch.cat([z, z[:, :1]], 1), scale_factor=2)


def stub_models(monkeypatch, train):
    monkeypatch.setattr(train, "embed_prompts", lambda prompts: torch.nn.functional.normalize(torch.randn(len(prompts), 768), dim=-1))
    monkeypatch.setattr(train, "load_decoder", lambda device: StubDecoder())


def test_train_writes_logs_eval_grid_and_loadable_checkpoint(train, tmp_path, monkeypatch):
    stub_models(monkeypatch, train)
    train.main(str(write_config(tmp_path, steps=3, eval_every=2, checkpoint_every=1)))
    run = tmp_path / "runs" / "smoke"

    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3]
    assert all(len(r["loss_by_t"]) == 10 and r["loss"] > 0 and r["grad_norm"] > 0 and r["imgs_per_s"] > 0 for r in log)
    # step 1 sees exactly zero matrix gradients (the final linear starts at zero), so capture is 0 there
    assert all(0 <= r["usui"]["grad_capture"] <= 1 for r in log) and log[2]["usui"]["grad_capture"] > 0
    assert "tangent_live_fraction" in log[2]["usui"]
    assert (run / "config.toml").exists() and (run / "eval_prompts.txt").read_text() == "a red apple\na blue car\n"
    assert log[0]["lr_usui"] == 3e-3 and log[0]["lr_fallback"] == 1e-3

    evals = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [2, 3]  # every eval_every, and the last step
    from PIL import Image
    assert Image.open(evals[0]["image"]).size == (3 * 32, 2 * 32)  # seeds across, prompts down

    checkpoint = run / "checkpoints" / "step_0000003"
    first = load_file(run / "checkpoints" / "step_0000001" / "model.safetensors")
    last = load_file(checkpoint / "model.safetensors")
    assert not torch.equal(first["blocks.0.qkv.weight"], last["blocks.0.qkv.weight"])  # a UsuiTrack matrix moved

    restored = DiT(torch.zeros(768), depth=2)
    restored.load_state_dict(load_file(checkpoint / "model.safetensors"))
    assert restored.final.weight.abs().max() > 0  # trained, not the zero init
    usui_state, adam_state = torch.load(checkpoint / "optimizer.pt")
    matrix, fallback = build_optimizers(restored, 3e-3, 0.95, 1e-3)
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
