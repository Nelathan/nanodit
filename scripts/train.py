"""Train the DiT on precomputed shards.

usage: train.py CONFIG.toml
"""
import glob
import json
import math
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from tinydit.data import Batcher, load_shards
from tinydit.model import DiT
from tinydit.objective import flow_loss, loss_by_t_bucket
from tinydit.optim import build_optimizers
from tinydit.sampler import euler_sample

PROMPT_PREFIX = "title: none | text: "  # SPEC §1
T_BUCKETS = 10


@dataclass(frozen=True)
class Config:
    name: str
    runs_dir: str
    resolution: int
    parquet_glob: str
    null_embedding: str
    depth: int
    batch: int
    steps: int
    seed: int
    usui_lr: float
    beta: float
    fallback_lr: float
    shift: float  # timestep shift, PLAN C3
    checkpoint_every: int
    eval_every: int
    eval_prompts: str
    eval_seeds: list[int]
    guidance: float
    sampler_steps: int


def embed_prompts(prompts: list[str]) -> torch.Tensor:
    """[P, 768] EmbeddingGemma 2 embeddings (unit norm from the sentence-transformers pipeline), then freed."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("google/embeddinggemma-2", config_kwargs=dict(vision_config=None, audio_config=None), model_kwargs=dict(dtype=torch.float32), device="cpu")
    return encoder.encode([PROMPT_PREFIX + p for p in prompts], convert_to_tensor=True)


def load_decoder(device: torch.device):
    from diffusers import AutoModel

    return AutoModel.from_pretrained("ostris/primal_vae", variant="2x2", trust_remote_code=True, torch_dtype=torch.float32).to(device).eval()


@torch.no_grad()
def evaluate(model, decoder, text, cfg: Config, step: int, run_dir: Path, device: torch.device) -> None:
    """Fixed prompts x fixed seeds: one PNG grid, a row per prompt and a column per seed."""
    side = cfg.resolution // 2
    noise = torch.stack([torch.randn(2, side, side, generator=torch.Generator().manual_seed(seed)) for seed in cfg.eval_seeds])
    n_prompts, n_seeds = len(text), len(noise)
    noise = noise.repeat(n_prompts, 1, 1, 1).to(device)  # row-major: prompt p, seed s at p * n_seeds + s
    text = text.repeat_interleave(n_seeds, dim=0).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        latents = euler_sample(model, noise, text, cfg.sampler_steps, cfg.guidance)
    images = decoder.decode(latents.float()).clamp(-1, 1)
    images = ((images + 1) * 127.5).round().byte().cpu().numpy()  # [P*S, 3, res, res]
    grid = images.reshape(n_prompts, n_seeds, 3, cfg.resolution, cfg.resolution).transpose(0, 3, 1, 4, 2)
    path = run_dir / "eval" / f"step_{step:07d}.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(grid.reshape(n_prompts * cfg.resolution, n_seeds * cfg.resolution, 3)).save(path)
    append_jsonl(run_dir / "eval.jsonl", dict(step=step, image=str(path), guidance=cfg.guidance, sampler_steps=cfg.sampler_steps, seeds=cfg.eval_seeds))


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def save_checkpoint(model, optimizers, step: int, run_dir: Path) -> None:
    path = run_dir / "checkpoints" / f"step_{step:07d}"
    path.mkdir(parents=True)
    save_file(model.state_dict(), path / "model.safetensors")
    torch.save([o.state_dict() for o in optimizers], path / "optimizer.pt")


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = Config(**tomllib.load(f))
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.runs_dir) / cfg.name
    if run_dir.exists():
        raise FileExistsError(run_dir)  # never overwrite an earlier run

    latents, embeddings = load_shards(sorted(glob.glob(cfg.parquet_glob)), cfg.resolution)
    batcher = Batcher(latents, embeddings, cfg.batch, device, torch.float32)
    model = DiT(load_file(cfg.null_embedding)["null"], cfg.depth).to(device)
    optimizers = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.fallback_lr)
    prompts_path = Path(config_path).parent / cfg.eval_prompts
    eval_text, decoder = embed_prompts(prompts_path.read_text().splitlines()), load_decoder(device)

    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.toml")
    shutil.copy(prompts_path, run_dir / "eval_prompts.txt")

    for step in range(1, cfg.steps + 1):
        start = time.perf_counter()
        x, text = batcher.next()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, per_sample, t = flow_loss(model, x, text, cfg.shift)
        loss.backward()
        loss_value = loss.item()
        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]).item()
        # UsuiTrack zeroes non-finite gradients but the AdamW fallback does not: stop before either steps.
        if not (math.isfinite(loss_value) and math.isfinite(grad_norm)):
            raise RuntimeError(f"non-finite loss {loss_value} or gradient norm {grad_norm} at step {step}")
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
        buckets = loss_by_t_bucket(per_sample.detach(), t, T_BUCKETS).tolist()
        append_jsonl(run_dir / "train.jsonl", dict(
            step=step,
            loss=loss_value,
            loss_by_t=[None if v != v else v for v in buckets],
            lr_usui=optimizers[0].param_groups[0]["lr"],
            lr_fallback=optimizers[1].param_groups[0]["lr"],
            grad_norm=grad_norm,
            usui=optimizers[0].pop_diagnostics(),
            imgs_per_s=cfg.batch / (time.perf_counter() - start),
        ))
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(model, decoder, eval_text, cfg, step, run_dir, device)
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            save_checkpoint(model, optimizers, step, run_dir)


if __name__ == "__main__":
    main(sys.argv[1])
