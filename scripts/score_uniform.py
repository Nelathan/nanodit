"""Score a checkpoint of a run that predates the uniform score on the uniform t grid, both sets, as the training score
does (SPEC §8); appends to RUN_DIR/score_uniform.jsonl and RUN_DIR/score_lunara_uniform.jsonl. Run it with the code the
run trained with, the uniform grid's commit on top.

usage: score_uniform.py RUN_DIR STEP
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.objective import t_uniform
from train import SCORE_TS, Config, append_jsonl, build_model, score, score_rows


def main(run_dir: str, step: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    ts = t_uniform(SCORE_TS)
    for name, path, resolution in (("score_uniform", cfg.eval_set, cfg.resolution), ("score_lunara_uniform", cfg.lunara_set, cfg.lunara_resolution)):
        scores = score(model, *score_rows(path, resolution, cfg.eval_seed), ts, cfg.micro_batch, device)
        append_jsonl(run / f"{name}.jsonl", dict(step=step, t=ts.tolist()) | scores)
        print(f"{run.name} step {step}, {name}: loss {scores['loss']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
