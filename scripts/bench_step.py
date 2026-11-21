"""Training-step speed of a config's model on the rows of one Parquet at one resolution: forward + backward over the
micro-batches of a batch, then the UsuiTrack and fallback steps, as `train.py` steps without its records. For speed at
a resolution the training set does not hold yet (the Lunara eval set has 256 px latents). Rows are taken in file order,
cycling; the captions are the set's own, padded to the file's longest, not to each micro-batch's as in training,
and copied from the host each micro-batch: the speed compares across models, not to `train.py`'s img/s.

usage: bench_step.py CONFIG.toml ROWS.parquet RESOLUTION MICRO_BATCH
"""
import sys
import time
import tomllib

import torch

from nanodit.data import read_rows
from nanodit.objective import flow_loss
from nanodit.optim import build_optimizers
from train import Config, build_model, compile_blocks

WARMUP, STEPS = 10, 20


def main(config_path: str, rows_path: str, resolution: int, micro_batch: int) -> None:
    with open(config_path, "rb") as f:
        cfg = Config(**tomllib.load(f))
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16)
    usui, fallback = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.rank_fraction, cfg.fallback_lr)
    compile_blocks(model)
    x, prompt = read_rows(rows_path, resolution)
    accumulate = cfg.batch // micro_batch
    starts = iter(range(0, 10**9, micro_batch))

    def step() -> None:
        for _ in range(accumulate):
            i = next(starts) % (len(x) - micro_batch)
            loss, _, _ = flow_loss(model, x[i : i + micro_batch].to(device, torch.float32), prompt.rows(i, i + micro_batch).to(device, torch.float32), cfg.shift, cfg.t_sigma)
            (loss / accumulate).backward()
        for optimizer in (usui, fallback):
            optimizer.step()
            optimizer.zero_grad()

    for _ in range(WARMUP):  # shapes compiled, Usui's frames drawn
        step()
    usui.pop_diagnostics()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    whole = (time.perf_counter() - start) / STEPS
    print(f"{resolution} px, batch {cfg.batch} as {accumulate} x {micro_batch}: {1000 * whole:.1f} ms per step, {cfg.batch / whole:.1f} img/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
