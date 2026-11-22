"""Whether the optimizer's step carries the per-channel scale the loss asks of a branch's write. From a checkpoint and
its optimizer state, a few training steps on the training stream at the run's settings, its configured lr (the
run's own after warmup when it does not anneal) and the plain loss; each writer (`out` of a mixer,
`w_down` of the MLP) is wrapped as `W x · (1 + s)` with `s` zero and outside the optimizers, so -∂L/∂s_c is the
demand on row c's scale and the run trains as it would. Per step the step's row-scale component is read off the
weights, <ΔW_c, W_c> / |W_c|². Per writer and part (slice: the first TOKEN_DIM rows; meta: the rest), over channels:
the share of rows whose demand is persistent (|mean| over 3 standard errors); the demand's coherence,
|Σ_c mean| / Σ_c |mean|, 1 when one scalar could serve every row (for `w_down`, normed after, the demand sums to zero
over the width, so its coherence is the slice's against the meta part's); the demand's split-half correlation, its
reliability; the correlation of the step's row-scale component with the demand, summed over the same steps, which
shares their gradient noise, and across halves (one half's step against the other's demand, averaged both ways),
which needs a demand that persists; and the matrix's norm over its init norm. Nothing is saved.

usage: probe_write_scales.py RUN_DIR STEP STEPS
"""
import sys
import tomllib
from pathlib import Path

import torch
from torch import nn

from nanodit.data import Stream
from nanodit.model import TOKEN_DIM
from nanodit.optim import build_optimizers
from train import Config, accumulate_gradients, build_model, compile_blocks, load_checkpoint


class RowScaled(nn.Module):
    """`inner(x) · (1 + s)`: a zero row scale whose gradient is the loss's demand on each output row's scale."""

    def __init__(self, inner: nn.Linear):
        super().__init__()
        self.inner = inner
        self.s = nn.Parameter(torch.zeros(inner.out_features, device=inner.weight.device))  # fp32: its grad sums micro-batches

    def forward(self, x):
        y = self.inner(x)
        return (y.float() * (1 + self.s)).to(y.dtype)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def main(run_dir: str, step: int, steps: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16)
    stream = Stream(cfg.dataset, cfg.resolution, cfg.micro_batch, device, torch.float32, cfg.seed + 1)
    optimizers = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.rank_fraction, cfg.fallback_lr)
    load_checkpoint(model, optimizers, run / "checkpoints" / f"step_{step:07d}")
    writers = []  # (block, kind, writer name, wrapper)
    for i, block in enumerate(model.blocks):
        for name in ("out", "w_down"):
            wrapped = RowScaled(getattr(block, name))
            setattr(block, name, wrapped)
            writers.append((i, "conv" if block.is_conv else "attn", "mixer" if name == "out" else "MLP", wrapped))
    compile_blocks(model)

    demand = [[] for _ in writers]  # per writer, per step: -∂L/∂s [width]
    moved = [[] for _ in writers]  # per writer, per step: <ΔW_c, W_c> / |W_c|²
    for k in range(steps):
        before = [w.inner.weight.detach().float().clone() for *_, w in writers]
        accumulate_gradients(model, stream, cfg.batch // cfg.micro_batch, cfg.shift, cfg.t_sigma)
        for (*_, w), d in zip(writers, demand):
            d.append(-w.s.grad.float().clone())
            w.s.grad = None
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
        for (*_, w), b, m in zip(writers, before, moved):
            m.append(((w.inner.weight.detach().float() - b) * b).sum(1) / b.square().sum(1))
        print(f"step {k + 1}/{steps}", flush=True)

    print(f"{run.name} step {step}, {steps} steps: per writer and part, persistent share / coherence / split-half r / "
          "step-demand r, same steps / across halves, and the matrix's norm over its init norm")
    for (i, kind, writer, w), d, m in zip(writers, demand, moved):
        d, m = torch.stack(d), torch.stack(m)  # [steps, width]
        mean, sem = d.mean(0), d.std(0) / steps**0.5
        half = steps // 2
        ratio = w.inner.weight.float().norm() / optimizers[0].state[w.inner.weight]["init_norm"]
        reads = []
        for part in (slice(0, TOKEN_DIM), slice(TOKEN_DIM, None)):
            persistent = float((mean[part].abs() > 3 * sem[part]).float().mean())
            coherence = float(mean[part].sum().abs() / mean[part].abs().sum())
            across = (corr(m[:half, part].sum(0), d[half:, part].sum(0)) + corr(m[half:, part].sum(0), d[:half, part].sum(0))) / 2
            reads.append(f"{persistent:.2f} / {coherence:.2f} / {corr(d[:half, part].sum(0), d[half:, part].sum(0)):+.2f} / {corr(m[:, part].sum(0), d[:, part].sum(0)):+.2f} / {across:+.2f}")
        print(f"{i:2d} {kind} {writer:5s} slice {reads[0]}   meta {reads[1]}   norm {ratio:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
