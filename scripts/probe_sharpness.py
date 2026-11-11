"""The top Hessian eigenvalue of the loss at checkpoints, on one fixed batch (PLAN, edge of stability).

usage: probe_sharpness.py CONFIG.toml CHECKPOINT [CHECKPOINT ...]

CONFIG is a train.py config (model size, eval set, t shift and sigma). The batch is the first ROWS rows of its eval set
with their captions, one fixed noise and one fixed t each, so every checkpoint is read on the same loss. The Hessian-vector
product is the central difference of the gradient, `(g(W + eps v) - g(W - eps v)) / 2 eps`, in fp32; the eigenvalue is
the Rayleigh quotient after ITERATIONS power iterations (40 iterations agree with 12 to four digits at the
checkpoints read so far). Power iteration finds the eigenvalue of largest magnitude: read the sign. The loss is the
conditional one, without prompt dropout, so the value is the sharpness of this batch's loss. Prints one JSON line per checkpoint.
"""
import json
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.data import read_rows
from nanodit.objective import noisy, per_sample_loss, sample_t
from train import Config, build_model

ROWS = 128
CHUNK = 16  # rows per forward/backward
ITERATIONS = 12
FD_STEP = 1e-2  # along a unit-norm v over all parameters


def fixed_batch(cfg, device):
    x, prompt = read_rows(cfg.eval_set, cfg.resolution)
    x, prompt = x[:ROWS].float().to(device), prompt.rows(0, ROWS).to(device, torch.float32)
    torch.manual_seed(0)
    return x, prompt, sample_t(ROWS, cfg.shift, device, cfg.t_sigma), torch.randn_like(x)


def accumulate_gradient(model, batch, sign):
    """Adds `sign` times the gradient of the batch's mean loss to the parameters' .grad."""
    x, prompt, t, eps = batch
    for i in range(0, ROWS, CHUNK):
        s = slice(i, i + CHUNK)
        x_t = noisy(x[s], eps[s], t[s])
        loss = per_sample_loss(model(x_t, t[s], prompt.rows(i, i + CHUNK)), x[s], eps[s], x_t).sum() / ROWS
        (sign * loss).backward()


@torch.no_grad()
def translate(params, direction, scale):
    for p, d in zip(params, direction):
        p.add_(d, alpha=scale)


def hessian_vector(model, batch, params, direction):
    """H v into the parameters' .grad."""
    for p in params:
        p.grad = torch.zeros_like(p)  # a parameter outside the loss keeps a zero gradient
    with torch.no_grad():
        translate(params, direction, FD_STEP)
    accumulate_gradient(model, batch, 1.0)
    with torch.no_grad():
        translate(params, direction, -2 * FD_STEP)
    accumulate_gradient(model, batch, -1.0)
    with torch.no_grad():
        translate(params, direction, FD_STEP)
        for p in params:
            p.grad /= 2 * FD_STEP


def top_eigenvalue(model, batch):
    params = [p for p in model.parameters() if p.requires_grad]
    generator = torch.Generator(device=params[0].device).manual_seed(0)
    direction = [torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype) for p in params]
    norm = torch.sqrt(sum(d.square().sum() for d in direction))
    for d in direction:
        d /= norm
    eigenvalue = float("nan")
    for _ in range(ITERATIONS):
        hessian_vector(model, batch, params, direction)
        with torch.no_grad():
            eigenvalue = sum((d * p.grad).sum() for d, p in zip(direction, params)).item()
            norm = torch.sqrt(sum(p.grad.square().sum() for p in params))
            for d, p in zip(direction, params):
                d.copy_(p.grad / norm)
    return eigenvalue


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, torch.float32)
    batch = fixed_batch(cfg, device)
    for path in sys.argv[2:]:
        model.load_state_dict(load_file(Path(path) / "model.safetensors"))
        print(json.dumps(dict(checkpoint=path, eigenvalue=top_eigenvalue(model, batch))), flush=True)
