"""Where a training step's time goes: the step of `train.py` on a config's model and data, timed whole and by phase
(data, forward + backward, UsuiTrack step, fallback step), and a profiler table of one step's top operations.

usage: probe_step.py CONFIG.toml
"""
import sys
import time
import tomllib

import torch
from torch.profiler import ProfilerActivity, profile

from nanodit.data import Stream
from nanodit.objective import flow_loss
from nanodit.optim import build_optimizers
from train import Config, accumulate_gradients, build_model

WARMUP, STEPS, PROFILED = 40, 30, 3


def main(config_path: str) -> None:
    with open(config_path, "rb") as f:
        cfg = Config(**tomllib.load(f))
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16)
    stream = Stream(cfg.dataset, cfg.resolution, cfg.micro_batch, device, torch.float32, cfg.seed)
    usui, fallback = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.rank_fraction, cfg.fallback_lr)
    for block in model.blocks:
        block.compile()
    accumulate = cfg.batch // cfg.micro_batch

    def step() -> None:
        accumulate_gradients(model, stream, accumulate, cfg.shift, cfg.t_sigma)
        fallback_grads = [p.grad for group in fallback.param_groups for p in group["params"] if p.grad is not None]
        torch.nn.utils.get_total_norm(fallback_grads)
        for optimizer in (usui, fallback):
            optimizer.step()
            optimizer.zero_grad()

    for _ in range(WARMUP):  # every bucket's shapes compiled, Usui's frames drawn
        step()
    usui.pop_diagnostics()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    whole = (time.perf_counter() - start) / STEPS
    print(f"step, as train.py steps without the schedule and records: {1000 * whole:.1f} ms, {cfg.batch / whole:.0f} img/s", flush=True)

    phases = dict(data=0.0, forward_backward=0.0, usui_step=0.0, fallback_step=0.0)
    for _ in range(STEPS):
        batches = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for bucket in stream.draw(accumulate):
            x, prompt = bucket.next()
            batches.append((x.clone(), prompt))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for x, prompt in batches:
            loss, _, _ = flow_loss(model, x, prompt, cfg.shift, cfg.t_sigma)
            (loss / accumulate).backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        usui.step()
        usui.zero_grad()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        fallback.step()
        fallback.zero_grad()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        for name, (a, b) in zip(phases, [(t0, t1), (t1, t2), (t2, t3), (t3, t4)]):
            phases[name] += (b - a) / STEPS
    print("phases, each synchronised: " + ", ".join(f"{name} {1000 * v:.1f} ms" for name, v in phases.items()), f"(sum {1000 * sum(phases.values()):.1f})", flush=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILED):
            step()
        torch.cuda.synchronize()
    print(f"top operations over {PROFILED} steps, by self CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)
    print(f"top operations over {PROFILED} steps, by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
