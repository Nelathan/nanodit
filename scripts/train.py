"""Train the DiT on a shuffled Parquet of precomputed rows.

usage: train.py CONFIG.toml
"""
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

from nanodit.data import Stream, pad_sequences, read_rows
from nanodit.model import DiT, Prompt
from nanodit.objective import explained_variance, explained_variance_by_t_bucket, flow_loss, loss_by_t_bucket, noisy, per_sample_loss, t_quantiles
from nanodit.optim import build_optimizers
from nanodit.sampler import euler_sample

T_BUCKETS = 10
SCORE_TS = 10  # the score's t grid: equal-probability bins of the shift-1 logit-normal, whatever t the run trains on (SPEC §8)
SCORE_SHIFT = 1.0
WANDB_BUCKETS = 5  # wandb shows 5 buckets: the train.jsonl and score.jsonl series have 10


@dataclass(frozen=True)
class Config:
    name: str
    runs_dir: str
    resolution: int
    dataset: str  # one Parquet from scripts/shuffle.py
    null_embedding: str
    depth: int
    width: int  # the residual stream; heads of HEAD_DIM, the MLP hidden MLP_RATIO times it
    batch: int
    micro_batch: int  # batch is built by accumulating batch / micro_batch of these
    steps: int  # steps this run takes, after init_from's
    seed: int
    usui_lr: float  # Usui's relative step (SPEC §6)
    beta: float
    rank_fraction: float  # Usui's frame size (SPEC §6)
    norm_max: float  # each tracked matrix's norm is clipped at this multiple of its init norm (SPEC §6)
    fallback_lr: float
    warmup: int  # steps of linear lr warmup from lr / warmup, both optimizers
    shift: float  # timestep shift, PLAN C3
    log_every: int  # steps per train.jsonl record, averaged over them on the GPU
    checkpoint_every: int
    eval_every: int
    eval_prompts: str
    eval_seed: int  # prompt i's noise takes seed eval_seed + i
    guidance: float
    sampler_steps: int
    eval_set: str  # a precomputed Parquet never trained on, scored at every checkpoint
    wandb_project: str
    init_from: str | None = None  # a checkpoint directory: its weights and optimizer state, its step count continued
    t_sigma: float = 0.7  # std of the logit-normal t, SPEC §4
    conv_period: int = 4  # every conv_period-th block is attention, the rest are conv blocks; 1 is all attention
    cross: bool = False  # the attention blocks also read the caption's token states by cross attention
    pooled_ada: bool = True  # the pooled caption embedding conditions the adaLN
    lr_final_fraction: float = 1.0  # both lrs fall linearly over `steps` from the peak to this fraction of it


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype) -> DiT:
    """The config's DiT at its init, the null prompt from the precomputed empty caption."""
    null = load_file(cfg.null_embedding)
    return DiT(null["null"], null["null_tokens"], cfg.depth, cfg.width, cfg.conv_period, cfg.cross, cfg.pooled_ada).to(device, dtype)


def load_checkpoint(model, optimizers, path: Path) -> int:
    """Loads a `save_checkpoint` directory into a built model and its optimizers. Every optimizer setting but the lr
    comes from the checkpoint (beta, rank fraction, betas); the lr, the step rule and the norm pin are the configured
    ones, as the saved lr belongs to the earlier run's schedule and the other two define what `usui_lr` means. Returns the step it was saved at."""
    model.load_state_dict(load_file(path / "model.safetensors"))
    for optimizer, state in zip(optimizers, torch.load(path / "optimizer.pt", map_location="cpu"), strict=True):
        configured = [{key: group[key] for key in ("lr", "step_rule", "norm_bounds") if key in group} for group in optimizer.param_groups]
        optimizer.load_state_dict(state)
        for group, keep in zip(optimizer.param_groups, configured, strict=True):
            group.update(keep)
            group.pop("initial_lr", None)  # else the schedule built next would take the saved lr as its base
    return int(path.name.removeprefix("step_"))


def embed_prompts(prompts: list[str]) -> Prompt:
    """The prompts as EmbeddingGemma 2 gives them (SPEC §1): the unit-norm pooled embeddings and the 512-d token states, as in the precomputed rows."""
    from precompute import encode_text, load_text_encoder

    pooled, sequences = encode_text(load_text_encoder("cpu"), prompts)
    tokens, mask = pad_sequences(sequences)
    return Prompt(pooled, tokens, mask)


def load_decoder(device: torch.device):
    from diffusers import AutoModel

    return AutoModel.from_pretrained("ostris/primal_vae", variant="2x2", trust_remote_code=True, torch_dtype=torch.float32).to(device).eval()


@torch.no_grad()
def evaluate(model, decoder, prompt, cfg: Config, step: int, run_dir: Path, device: torch.device) -> None:
    """Fixed prompts, each with its own fixed noise: one PNG grid of the prompts in file order, row-major, as square as fits."""
    side = cfg.resolution // 2
    n_prompts = len(prompt.pooled)
    noise = torch.cat([torch.randn(1, 2, side, side, generator=torch.Generator().manual_seed(cfg.eval_seed + i)) for i in range(n_prompts)])
    latents = euler_sample(model, noise.to(device), prompt.to(device, torch.float32), cfg.sampler_steps, cfg.guidance)
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
    append_jsonl(run_dir / "eval.jsonl", dict(step=step, image=str(path), guidance=cfg.guidance, sampler_steps=cfg.sampler_steps, seeds=list(range(cfg.eval_seed, cfg.eval_seed + n_prompts))))
    wandb.log({"eval": wandb.Image(str(path), caption="prompts row-major in eval_prompts.txt order")}, step=step)


@torch.no_grad()
def score(model, x, prompt, noise, ts, batch: int, device: torch.device) -> dict:
    """The loss on fixed rows, each with its caption and fixed noise, at every t of a fixed grid: one number
    comparable across steps and runs, and its value at each t. The same for R² (`explained_variance`), per t and
    over the grid. Keys keep the `_caption` suffix of the records that also scored the null embedding."""
    energy, per_t = x.float().square().mean().item(), []
    for t_value in ts.tolist():
        total = 0.0
        for i in range(0, len(x), batch):
            xb, eb = x[i : i + batch].to(device, torch.float32), noise[i : i + batch].to(device)
            pb = prompt.rows(i, i + batch).to(device, torch.float32)
            t = torch.full((len(xb),), t_value, device=device)
            x_t = noisy(xb, eb, t)
            total += per_sample_loss(model(x_t, t, pb), xb, eb, x_t).sum()
        per_t.append(float(total) / len(x))
    errors = [loss * t_value**2 for loss, t_value in zip(per_t, ts.tolist())]
    return dict(
        loss_caption=sum(per_t) / len(per_t),
        loss_caption_by_t=per_t,
        r2_caption=1 - sum(errors) / len(errors) / energy,
        r2_caption_by_t=[1 - error / energy for error in errors],
    )


@torch.no_grad()
def weight_reads(model, matrix) -> dict:
    """The tracked matrices' global Frobenius norm and its ratio to the init norm, and per role the stable rank
    ||W||_F^2 / sigma_max^2 over the shorter side, averaged over the role's blocks (SPEC §8). A role is the
    parameter's name without its block index, prefixed by the block's mixer, since conv and attention both have `out`."""
    names = {id(p): n for n, p in model.named_parameters()}
    kinds = {f"blocks.{i}.": "conv." if block.is_conv else "attn." for i, block in enumerate(model.blocks)}
    total, init, ranks = 0.0, 0.0, {}
    for group in matrix.param_groups:
        for param in group["params"]:
            weight, name = param.float(), names[id(param)].removesuffix(".weight")
            frobenius = weight.square().sum()
            total += frobenius.item()
            init += matrix.state[param]["init_norm"] ** 2
            stable = (frobenius / torch.linalg.matrix_norm(weight, ord=2).square()).item() / min(weight.shape)
            prefix = next((p for p in kinds if name.startswith(p)), "")
            role = kinds[prefix] + name.removeprefix(prefix) if prefix else name
            ranks.setdefault(role, []).append(stable)
    return dict(norm=math.sqrt(total), norm_ratio=math.sqrt(total / init), stable_rank={role: sum(v) / len(v) for role, v in ranks.items()})


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def bucket_label(i: int) -> str:
    return f"{i / WANDB_BUCKETS:.1f}-{(i + 1) / WANDB_BUCKETS:.1f}"


def wandb_scalars(record: dict, by_t: dict) -> dict:
    """A train.jsonl record, flat, under `train/`, with `by_t` (the interval's loss and R² in the wandb buckets)
    as one group per topic, `train_<topic>_by_t/t<lo>-<hi>`, and Usui's reads under `usuitrack/`, as ai-toolkit
    names them. wandb groups on the first slash only, so a group holds its series one level deep."""
    scalars = {f"train/{key}": value for key, value in record.items() if key not in ("step", "loss_by_t", "r2_by_t", "usui")}
    for name, values in by_t.items():
        scalars |= {f"train_{name}/t{bucket_label(i)}": v for i, v in enumerate(values) if v is not None}
    return scalars | {f"usuitrack/{key}": v for key, v in record["usui"].items()}


def wandb_scores(scores: dict) -> dict:
    """A score record, flat: the means under `eval/`, and each by-t series as the means of neighbouring grid points,
    the t's of one fifth of the grid's probability mass each, in one group per topic,
    `eval_<topic>_by_t/q<lo>-<hi>`, named by its probability-mass quantiles (not t)."""
    flat = {f"eval/{key}": scores[key] for key in ("loss_caption", "r2_caption")}
    size = SCORE_TS // WANDB_BUCKETS
    for key in ("loss_caption_by_t", "r2_caption_by_t"):
        flat |= {f"eval_{key}/q{bucket_label(i)}": sum(scores[key][i * size : (i + 1) * size]) / size for i in range(WANDB_BUCKETS)}
    return flat


def wandb_weights(weights: dict) -> dict:
    return {"weights/norm": weights["norm"], "weights/norm_ratio": weights["norm_ratio"]} | {f"stable_rank/{role}": v for role, v in weights["stable_rank"].items()}


def save_checkpoint(model, optimizers, step: int, run_dir: Path) -> None:
    path = run_dir / "checkpoints" / f"step_{step:07d}"
    path.mkdir(parents=True)
    save_file(model.state_dict(), path / "model.safetensors")
    torch.save([o.state_dict() for o in optimizers], path / "optimizer.pt")


class Interval:
    """The training reads of the steps since the last record, kept on the GPU; reading them is the one
    device-to-host transfer per record."""

    def __init__(self):
        self.losses, self.per_samples, self.ts, self.energies, self.grad_norms = [], [], [], [], []
        self.start = time.perf_counter()

    def add(self, loss, per_sample, t, energy, grad_norm) -> None:
        self.losses.append(loss), self.per_samples.append(per_sample), self.ts.append(t), self.energies.append(energy), self.grad_norms.append(grad_norm)

    def by_t(self, buckets: int) -> dict:
        """The interval's loss and R² in `buckets` equal-width t bins; None where a bin is empty."""
        per_sample, t, energy = torch.cat(self.per_samples), torch.cat(self.ts), torch.cat(self.energies)
        return dict(
            loss_by_t=[None if v != v else v for v in loss_by_t_bucket(per_sample, t, buckets).tolist()],
            r2_by_t=[None if v != v else v for v in explained_variance_by_t_bucket(per_sample, t, energy, buckets).tolist()],
        )

    def report(self, step: int, batch: int) -> dict:
        """Means over the interval. A non-finite step stops the run here: UsuiTrack zeroes non-finite
        gradients, but the AdamW fallback has stepped on them since."""
        losses, grad_norms = torch.stack(self.losses).tolist(), torch.stack(self.grad_norms).tolist()
        first = step - len(losses) + 1
        for i, (loss, grad_norm) in enumerate(zip(losses, grad_norms)):
            if not (math.isfinite(loss) and math.isfinite(grad_norm)):
                raise RuntimeError(f"non-finite loss {loss} or fallback gradient norm {grad_norm} at step {first + i}")
        per_sample, t, energy = torch.cat(self.per_samples), torch.cat(self.ts), torch.cat(self.energies)
        return dict(
            step=step,
            loss=sum(losses) / len(losses),
            r2=explained_variance(per_sample, t, energy).item(),
            **self.by_t(T_BUCKETS),
            fallback_grad_norm=sum(grad_norms) / len(grad_norms),
            imgs_per_s=batch * len(losses) / (time.perf_counter() - self.start),
        )


def accumulate_gradients(model, stream: Stream, micro_batches: int, shift: float, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward over `micro_batches` micro-batches without stepping. The fallback parameters end with the
    mean gradient over all of them (UsuiTrack ignores gradient scale). Returns the mean flow loss and the
    per-sample losses, t and clean-latent energy (mean square) of every sample."""
    losses, per_samples, ts, energies = [], [], [], []
    for _ in range(micro_batches):
        x, prompt = stream.next()
        loss, per_sample, t = flow_loss(model, x, prompt, shift, sigma)
        (loss / micro_batches).backward()
        losses.append(loss.detach()), per_samples.append(per_sample.detach()), ts.append(t), energies.append(x.square().flatten(1).mean(1))
    return torch.stack(losses).mean(), torch.cat(per_samples), torch.cat(ts), torch.cat(energies)


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
    stream = Stream(cfg.dataset, cfg.resolution, cfg.micro_batch, device, torch.float32, cfg.seed, sequences=cfg.cross)
    # Pure bf16: the fp32 init is rounded once, and the optimizers update with stochastic rounding.
    model = build_model(cfg, device, torch.bfloat16)
    optimizers = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.rank_fraction, cfg.norm_max, cfg.fallback_lr)
    start = load_checkpoint(model, optimizers, Path(cfg.init_from)) if cfg.init_from else 0
    for block in model.blocks:  # after Molegrad has swapped in its Linears, so the compiled graphs trace them
        block.compile()

    def lr_factor(done: int) -> float:
        return min(1.0, (done + 1) / max(1, cfg.warmup)) * (1 + (cfg.lr_final_fraction - 1) * done / cfg.steps)

    schedules = [torch.optim.lr_scheduler.LambdaLR(o, lr_factor) for o in optimizers]
    prompts_path = Path(config_path).parent / cfg.eval_prompts
    eval_prompt, decoder = embed_prompts(prompts_path.read_text().splitlines()), load_decoder(device)
    score_x, score_prompt = read_rows(cfg.eval_set, cfg.resolution)
    score_noise = torch.randn(score_x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))  # one draw per row, shared by every t and condition
    score_ts = t_quantiles(SCORE_TS, SCORE_SHIFT)

    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.toml")
    shutil.copy(prompts_path, run_dir / "eval_prompts.txt")
    # Entity and key come from $WANDB_ENTITY and $WANDB_API_KEY.
    wandb.init(project=cfg.wandb_project, name=cfg.name, config=asdict(cfg), dir=run_dir)

    accumulate = cfg.batch // cfg.micro_batch
    interval = Interval()
    last = start + cfg.steps
    for step in range(start + 1, last + 1):
        loss, per_sample, t, energy = accumulate_gradients(model, stream, accumulate, cfg.shift, cfg.t_sigma)
        # Tracked matrices have no .grad under Molegrad: this is the fallback parameters' norm.
        fallback_grads = [p.grad for group in optimizers[1].param_groups for p in group["params"] if p.grad is not None]  # the null tokens have none without cross attention
        interval.add(loss, per_sample, t, energy, torch.nn.utils.get_total_norm(fallback_grads))
        lr_usui, lr_fallback = (o.param_groups[0]["lr"] for o in optimizers)
        for optimizer, schedule in zip(optimizers, schedules):
            optimizer.step()
            optimizer.zero_grad()
            schedule.step()
        if step % cfg.log_every == 0 or step == last:
            record = interval.report(step, cfg.batch) | dict(lr_usui=lr_usui, lr_fallback=lr_fallback, usui=optimizers[0].pop_diagnostics())
            append_jsonl(run_dir / "train.jsonl", record)
            wandb.log(wandb_scalars(record, interval.by_t(WANDB_BUCKETS)), step=step)
            interval = Interval()
        paused = time.perf_counter()
        if step % cfg.eval_every == 0 or step == last:
            evaluate(model, decoder, eval_prompt, cfg, step, run_dir, device)
            scores = score(model, score_x, score_prompt, score_noise, score_ts, cfg.micro_batch, device)
            weights = weight_reads(model, optimizers[0])
            append_jsonl(run_dir / "score.jsonl", dict(step=step, t=score_ts.tolist()) | scores | dict(weights=weights))
            wandb.log(wandb_scores(scores) | wandb_weights(weights), step=step)
        if step % cfg.checkpoint_every == 0 or step == last:
            save_checkpoint(model, optimizers, step, run_dir)
        interval.start += time.perf_counter() - paused  # img/s counts training time only
    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1])
