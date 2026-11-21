"""How much of the trunk's correction is a rescale of its input: at the score's 10 t, on the first rows of the
held-out set with the score's noise, the slope of the trunk's correction F − x_t regressed on x_t (pooled over rows,
tokens and channels), against the same slope of the true correction T − x_t, which is c(t) − 1 for the data's own
variance; each one's linear share of its energy; and the loss against the loss of the best linear predictor c(t)·x_t.
CPU, fp32: it runs beside a training run.

usage: probe_linear_skip.py RUN_DIR STEP ROWS
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.data import read_rows
from nanodit.objective import noisy, t_quantiles, target
from train import SCORE_SHIFT, SCORE_TS, Config, build_model


def slope(y: torch.Tensor, x: torch.Tensor) -> float:
    return ((y * x).sum() / x.square().sum()).item()


@torch.no_grad()
def main(run_dir: str, step: int, rows: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    model = build_model(cfg, torch.device("cpu"), torch.float32).requires_grad_(False)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    x, prompt = read_rows(cfg.eval_set, cfg.resolution)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))[:rows]  # the score's draw over the whole set
    x, prompt = x[:rows].float(), prompt.rows(0, rows).to(torch.device("cpu"), torch.float32)
    print(f"{run.name} step {step}, {rows} held-out rows, latent mean square {x.square().mean():.3f}")
    print("t     | slope true (c−1)  model | linear share true  model | loss model  linear  identity")
    for t in t_quantiles(SCORE_TS, SCORE_SHIFT):
        x_t = noisy(x, noise, t.expand(rows))
        T = target(x, noise, x_t)
        F = model(x_t, t.expand(rows), prompt)
        true, made = T - x_t, F - x_t
        a_true, a_made = slope(true, x_t), slope(made, x_t)
        share_true = a_true**2 * x_t.square().sum() / true.square().sum()
        share_made = a_made**2 * x_t.square().sum() / made.square().sum()
        loss = (F - T).square().mean()
        linear = (T - (1 + a_true) * x_t).square().mean()
        print(f"{t:.2f}  | {a_true:+17.3f}  {a_made:+5.3f} | {share_true:17.3f}  {share_made:5.3f} | "
              f"{loss:10.4f}  {linear:6.4f}  {true.square().mean():8.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
