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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
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
    micro_batch: int  # batch is built by accumulating batch / micro_batch of these
    steps: int
    seed: int
    usui_lr: float  # Usui's fixed step (SPEC §6)
    beta: float
    rank_fraction: float  # Usui's frame size; the final linear takes the largest (SPEC §6)
    fallback_lr: float
    warmup: int  # steps of linear lr warmup from lr / warmup, both optimizers
    shift: float  # timestep shift, PLAN C3
    log_every: int  # steps per train.jsonl record, averaged over them on the GPU
    checkpoint_every: int
    eval_every: int
    eval_prompts: str
    eval_seed: int
    guidance: float
    sampler_steps: int
    wandb_project: str


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
    """Fixed prompts, one fixed noise: one PNG grid of the prompts in file order, row-major, as square as fits."""
    side = cfg.resolution // 2
    n_prompts = len(text)
    noise = torch.randn(1, 2, side, side, generator=torch.Generator().manual_seed(cfg.eval_seed)).repeat(n_prompts, 1, 1, 1)
    latents = euler_sample(model, noise.to(device), text.to(device), cfg.sampler_steps, cfg.guidance)
    images = decoder.decode(latents.float()).clamp(-1, 1)
    images = ((images + 1) * 127.5).round().byte().cpu().numpy()  # [P, 3, res, res]
    cols = math.ceil(math.sqrt(n_prompts))
    rows = math.ceil(n_prompts / cols)
    cells = np.zeros((rows * cols, 3, cfg.resolution, cfg.resolution), np.uint8)  # black past the last prompt
    cells[:n_prompts] = images
    grid = cells.reshape(rows, cols, 3, cfg.resolution, cfg.resolution).transpose(0, 3, 1, 4, 2)
    path = run_dir / "eval" / f"step_{step:07d}.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(grid.reshape(rows * cfg.resolution, cols * cfg.resolution, 3)).save(path)
    append_jsonl(run_dir / "eval.jsonl", dict(step=step, image=str(path), guidance=cfg.guidance, sampler_steps=cfg.sampler_steps, seed=cfg.eval_seed))
    wandb.log({"eval": wandb.Image(str(path), caption="prompts row-major in eval_prompts.txt order")}, step=step)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def wandb_scalars(record: dict) -> dict:
    """A train.jsonl record, flat: one series per t bucket, named by its lower edge, and Usui's reads under
    `usuitrack/`, as ai-toolkit names them."""
    scalars = {key: value for key, value in record.items() if key not in ("step", "loss_by_t", "usui")}
    scalars |= {f"loss_by_t/{i / T_BUCKETS:.1f}": v for i, v in enumerate(record["loss_by_t"]) if v is not None}
    return scalars | {f"usuitrack/{key}": v for key, v in record["usui"].items()}


def save_checkpoint(model, optimizers, step: int, run_dir: Path) -> None:
    path = run_dir / "checkpoints" / f"step_{step:07d}"
    path.mkdir(parents=True)
    save_file(model.state_dict(), path / "model.safetensors")
    torch.save([o.state_dict() for o in optimizers], path / "optimizer.pt")


class Interval:
    """The training reads of the steps since the last record, kept on the GPU; reading them is the one
    device-to-host transfer per record."""

    def __init__(self):
        self.losses, self.per_samples, self.ts, self.grad_norms = [], [], [], []
        self.start = time.perf_counter()

    def add(self, loss, per_sample, t, grad_norm) -> None:
        self.losses.append(loss), self.per_samples.append(per_sample), self.ts.append(t), self.grad_norms.append(grad_norm)

    def report(self, step: int, batch: int) -> dict:
        """Means over the interval. A non-finite step stops the run here: UsuiTrack zeroes non-finite
        gradients, but the AdamW fallback has stepped on them since."""
        losses, grad_norms = torch.stack(self.losses).tolist(), torch.stack(self.grad_norms).tolist()
        first = step - len(losses) + 1
        for i, (loss, grad_norm) in enumerate(zip(losses, grad_norms)):
            if not (math.isfinite(loss) and math.isfinite(grad_norm)):
                raise RuntimeError(f"non-finite loss {loss} or fallback gradient norm {grad_norm} at step {first + i}")
        buckets = loss_by_t_bucket(torch.cat(self.per_samples), torch.cat(self.ts), T_BUCKETS).tolist()
        return dict(
            step=step,
            loss=sum(losses) / len(losses),
            loss_by_t=[None if v != v else v for v in buckets],
            fallback_grad_norm=sum(grad_norms) / len(grad_norms),
            imgs_per_s=batch * len(losses) / (time.perf_counter() - self.start),
        )


def accumulate_gradients(model, batcher: Batcher, micro_batches: int, shift: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward over `micro_batches` micro-batches without stepping. The fallback parameters end with the
    mean gradient over all of them (UsuiTrack ignores gradient scale). Returns the mean loss and the
    per-sample losses and t of every sample."""
    losses, per_samples, ts = [], [], []
    for _ in range(micro_batches):
        x, text = batcher.next()
        loss, per_sample, t = flow_loss(model, x, text, shift)
        (loss / micro_batches).backward()
        losses.append(loss.detach()), per_samples.append(per_sample.detach()), ts.append(t)
    return torch.stack(losses).mean(), torch.cat(per_samples), torch.cat(ts)


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = Config(**tomllib.load(f))
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.runs_dir) / cfg.name
    if run_dir.exists():
        raise FileExistsError(run_dir)  # never overwrite an earlier run

    if cfg.batch % cfg.micro_batch:
        raise ValueError(f"batch {cfg.batch} is not a multiple of micro_batch {cfg.micro_batch}")
    # a record checks for non-finite steps, so nothing is evaluated or saved past an unchecked one
    if cfg.eval_every % cfg.log_every or cfg.checkpoint_every % cfg.log_every:
        raise ValueError(f"eval_every {cfg.eval_every} and checkpoint_every {cfg.checkpoint_every} must be multiples of log_every {cfg.log_every}")
    latents, embeddings = load_shards(sorted(glob.glob(cfg.parquet_glob)), cfg.resolution)
    batcher = Batcher(latents, embeddings, cfg.micro_batch, device, torch.float32)
    # Pure bf16: the fp32 init is rounded once, and the optimizers update with stochastic rounding.
    model = DiT(load_file(cfg.null_embedding)["null"], cfg.depth).to(device, torch.bfloat16)
    optimizers = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.rank_fraction, cfg.fallback_lr)
    schedules = [torch.optim.lr_scheduler.LambdaLR(o, lambda done: min(1.0, (done + 1) / max(1, cfg.warmup))) for o in optimizers]
    prompts_path = Path(config_path).parent / cfg.eval_prompts
    eval_text, decoder = embed_prompts(prompts_path.read_text().splitlines()), load_decoder(device)

    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.toml")
    shutil.copy(prompts_path, run_dir / "eval_prompts.txt")
    # Entity and key come from $WANDB_ENTITY and $WANDB_API_KEY.
    wandb.init(project=cfg.wandb_project, name=cfg.name, config=asdict(cfg), dir=run_dir)

    accumulate = cfg.batch // cfg.micro_batch
    interval = Interval()
    for step in range(1, cfg.steps + 1):
        loss, per_sample, t = accumulate_gradients(model, batcher, accumulate, cfg.shift)
        # Tracked matrices have no .grad under Molegrad: this is the fallback parameters' norm.
        fallback_grads = [p.grad for group in optimizers[1].param_groups for p in group["params"]]
        interval.add(loss, per_sample, t, torch.nn.utils.get_total_norm(fallback_grads))
        lr_usui, lr_fallback = (o.param_groups[0]["lr"] for o in optimizers)
        for optimizer, schedule in zip(optimizers, schedules):
            optimizer.step()
            optimizer.zero_grad()
            schedule.step()
        if step % cfg.log_every == 0 or step == cfg.steps:
            record = interval.report(step, cfg.batch) | dict(lr_usui=lr_usui, lr_fallback=lr_fallback, usui=optimizers[0].pop_diagnostics())
            append_jsonl(run_dir / "train.jsonl", record)
            wandb.log(wandb_scalars(record), step=step)
            interval = Interval()
        paused = time.perf_counter()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(model, decoder, eval_text, cfg, step, run_dir, device)
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            save_checkpoint(model, optimizers, step, run_dir)
        interval.start += time.perf_counter() - paused  # img/s counts training time only
    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1])
