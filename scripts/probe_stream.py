"""The residual stream's size through depth: after the patch embed and after each block, the RMS of the slice (the
first TOKEN_DIM channels, which start as x_t and end as the output) and of the meta channels, at the score's 10 t on
the first rows of the held-out set with the score's noise; the meta part split into its mean over an image's tokens
(the common mode) and the rest. STEP 0 reads the model at its init. CPU, fp32: it runs beside a training run.

usage: probe_stream.py RUN_DIR STEP ROWS
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.data import read_rows
from nanodit.model import TOKEN_DIM, patchify
from nanodit.objective import noisy, t_quantiles
from train import SCORE_SHIFT, SCORE_TS, Config, build_model


def rms(x: torch.Tensor) -> float:
    return x.square().mean().sqrt().item()


@torch.no_grad()
def main(run_dir: str, step: int, rows: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, torch.device("cpu"), torch.float32).requires_grad_(False)
    if step:
        model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    x, prompt = read_rows(cfg.eval_set, cfg.resolution)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))[:rows]  # the score's draw over the whole set
    x, prompt = x[:rows].float(), prompt.rows(0, rows).to(torch.device("cpu"), torch.float32)
    reads = {}  # stage name -> per t (slice, meta, meta common mode, meta rest)
    hooks = [model.patch_embed.register_forward_hook(lambda m, i, o: reads.setdefault("embed", []).append(torch.cat([i[0], o], -1)))]
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(lambda m, i_, o, i=i: reads.setdefault(i, []).append(o)))
    ts = t_quantiles(SCORE_TS, SCORE_SHIFT)
    for t in ts:
        model(noisy(x, noise, t.expand(rows)), t.expand(rows), prompt)
    for h in hooks:
        h.remove()
    target_rms = [rms(patchify((2 - t) * x - (1 - t) * noise)[0]) for t in ts]
    print(f"{run.name} step {step}, {rows} held-out rows; columns per t: slice RMS / meta RMS / meta common-mode RMS / meta rest RMS")
    print("t      " + "  ".join(f"{t:>23.2f}" for t in ts))
    print("target " + "  ".join(f"{v:23.3f}" for v in target_rms))
    for name, outs in reads.items():
        cells = []
        for o in outs:
            meta = o[..., TOKEN_DIM:]
            common = meta.mean(1, keepdim=True)
            cells.append(f"{rms(o[..., :TOKEN_DIM]):.2f}/{rms(meta):.2f}/{rms(common):.2f}/{rms(meta - common):.2f}")
        label = name if name == "embed" else f"{name:2d} {'conv' if model.blocks[name].is_conv else 'attn'}"
        print(f"{label:7s}" + "  ".join(f"{c:>23s}" for c in cells), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
