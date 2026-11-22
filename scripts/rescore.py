"""Score a run's checkpoint again as training scores it, on its held-out and Lunara sets on both t grids, under the
code it runs from: a score function or weight changed since the run (another decoder floor) applied to an old
checkpoint. Appends to RUN_DIR/{score,score_lunara,score_uniform,score_lunara_uniform}_NAME.jsonl.

usage: rescore.py RUN_DIR STEP NAME
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.objective import t_quantiles, t_uniform
from train import SCORE_SHIFT, SCORE_TS, Config, append_jsonl, build_model, score, score_rows


def main(run_dir: str, step: int, name: str) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    sets = {"score": score_rows(cfg.eval_set, cfg.resolution, cfg.eval_seed), "score_lunara": score_rows(cfg.lunara_set, cfg.lunara_resolution, cfg.eval_seed)}
    for grid, ts in (("", t_quantiles(SCORE_TS, SCORE_SHIFT)), ("_uniform", t_uniform(SCORE_TS))):
        for set_name, rows in sets.items():
            scores = score(model, *rows, ts, cfg.micro_batch, device)
            append_jsonl(run / f"{set_name}{grid}_{name}.jsonl", dict(step=step, t=ts.tolist()) | scores)
            print(f"{run.name} step {step}, {set_name}{grid}: loss {scores['loss']:.4f}, decoder {scores['loss_decoder']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
