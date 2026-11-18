"""Where in the network the prediction forms: the image channels of the residual stream read after every sublayer's
write, as a partial prediction F_k (the read-out applied early).

Two reads of a run's checkpoint:
- On the eval set, at the score's 10 t, per write: the loss of F_k, the RMS of the write on the image channels, and
  the energy of the clean estimate's error x0_k - x = t·(F_k - T) in three bands of spatial frequency on the 64 x 64
  latent: coarser than two tokens (period > 16 latent px), one to two tokens (8 to 16 px), and inside one token
  (< 8 px), with the clean latents' energy in each band, so error over energy is the share of a band not recovered.
- On the eval prompts, sampled as the training eval samples them (Euler, CFG, the prompts' fixed noise): at four
  steps of the trajectory, the clean estimate x0_k = (1 - t)·x_t + t·F_k decoded after every write, one PNG per
  step and prompt. Rows are the attention periods, columns the writes in order (conv blocks: mixer, which writes only
  the meta channels, then MLP; the attention block: self-attention, cross attention, MLP). And per prompt the clean
  estimate of the whole network at every sampler step, to see at which t the forms are fixed.

The writes are taken by hooks on each branch norm's output, times its gate, and re-added in bf16 in the model's order;
the sum is checked against each block's output.

usage: probe_residual_writes.py RUN_DIR STEP OUT_DIR [ROWS]
"""
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from nanodit.data import read_rows
from nanodit.model import TOKEN_DIM, Prompt, unpatchify
from nanodit.objective import noisy, per_sample_loss, t_quantiles, target
from nanodit.sampler import euler_sample
from train import SCORE_SHIFT, SCORE_TS, Config, build_model, embed_prompts, load_decoder

BATCH = 16
SAMPLE_STEPS = (3, 6, 8, 15)  # of the 30 Euler steps: t = 0.9, 0.8, 0.73, 0.5
SAMPLE_PROMPTS = (1, 6, 9, 10, 14)  # eval prompt indices: landscape, chess, beach, cabin, sign
BANDS = ((0, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 0.75))  # radial cycles per latent pixel; 0.75 > the corner's 0.707


class WriteCapture:
    """Hooks that record, per block, the residual after each write as the model adds it."""

    def __init__(self, model):
        self.model, self.mod, self.blocks = model, None, []
        model.ada.register_forward_hook(lambda m, i, out: setattr(self, "mod", out.view(out.shape[0], 6, -1)))
        for b, block in enumerate(model.blocks):
            block.register_forward_pre_hook(lambda m, args, b=b: self._start(b, args[0]))
            names = ["conv_norm"] if block.is_conv else ["attn_norm", "cross_norm"]
            for name in [*names, "mlp_norm"]:
                getattr(block, name).register_forward_hook(lambda m, i, out, b=b, name=name: self._write(b, name, out))
            block.register_forward_hook(lambda m, args, out, b=b: self._check(b, out))

    def reset(self) -> None:
        self.blocks = []

    def _start(self, b: int, x) -> None:
        self.blocks.append(dict(x=x, writes=[]))

    def _write(self, b: int, name: str, out) -> None:
        block, record = self.model.blocks[b], self.blocks[-1]
        mod = self.mod + block.table
        if name == "conv_norm":
            write = F.pad(mod[:, 2, None, TOKEN_DIM:] * out, (TOKEN_DIM, 0))
        elif name == "attn_norm":
            write = mod[:, 2, None] * out
        elif name == "cross_norm":
            write = block.cross_gate * out
        else:
            write = mod[:, 5, None] * out
        record["x"] = record["x"] + write
        record["writes"].append((name.removesuffix("_norm"), record["x"][..., :TOKEN_DIM].float(), write[..., :TOKEN_DIM].float()))

    def _check(self, b: int, out) -> None:
        assert torch.equal(self.blocks[-1]["x"], out), f"block {b}: the re-added writes do not give the block's output"

    def partials(self, h: int, w: int) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """(label, F_k [B, 2, H, W], write on the image channels [B, 2, H, W]) after every write, in order."""
        return [(f"{b}.{name}", unpatchify(res, h, w), unpatchify(write, h, w))
                for b, record in enumerate(self.blocks) for name, res, write in record["writes"]]


def band_energy(e: torch.Tensor) -> torch.Tensor:
    """Mean square of e [B, C, H, W] in each band of BANDS, summed over bands equal to the mean square of e: [B, len(BANDS)]."""
    H, W = e.shape[-2:]
    spectrum = torch.fft.fft2(e).abs().square().sum(1) / (H * W) ** 2 / e.shape[1]  # Parseval: sums to mean square
    fy, fx = torch.fft.fftfreq(H, device=e.device), torch.fft.fftfreq(W, device=e.device)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    return torch.stack([spectrum[:, (radius >= lo) & (radius < hi)].sum(-1) for lo, hi in BANDS], -1)


@torch.no_grad()
def eval_reads(model, capture, x, prompt, noise, ts, device) -> list[dict]:
    reads = []
    for t_value in ts.tolist():
        sums = {}
        for i in range(0, len(x), BATCH):
            xb, eb = x[i : i + BATCH].to(device, torch.float32), noise[i : i + BATCH].to(device)
            t = torch.full((len(xb),), t_value, device=device)
            x_t = noisy(xb, eb, t)
            capture.reset()
            model(x_t, t, prompt.rows(i, i + BATCH).to(device, torch.float32))
            T = target(xb, eb, x_t)
            rows = [("input", x_t, torch.zeros_like(x_t))] + capture.partials(x.shape[-2] // 8, x.shape[-1] // 8)
            for label, Fk, write in rows:
                s = sums.setdefault(label, dict(loss=0.0, write_rms=0.0, error_bands=torch.zeros(len(BANDS), device=device)))
                s["loss"] += per_sample_loss(Fk, xb, eb, x_t).sum().item()
                s["write_rms"] += write.square().flatten(1).mean(1).sqrt().sum().item()
                s["error_bands"] += band_energy(t_value * (Fk - T)).sum(0)
            clean = sums.setdefault("clean", dict(bands=torch.zeros(len(BANDS), device=device)))
            clean["bands"] += band_energy(xb).sum(0)
        n = len(x)
        reads.append(dict(t=t_value, clean_bands=(sums.pop("clean")["bands"] / n).tolist(),
                          writes=[dict(label=k, loss=v["loss"] / n, write_rms=v["write_rms"] / n, error_bands=(v["error_bands"] / n).tolist())
                                  for k, v in sums.items()]))
    return reads


@torch.no_grad()
def sample_reads(model, capture, decoder, prompt: Prompt, cfg: Config, out_dir: Path, device) -> None:
    """The eval sampler run as is; a pre-hook on the model catches x_t and t at every step, the writes at the chosen ones."""
    side, steps = cfg.resolution // 2, cfg.sampler_steps
    noise = torch.cat([torch.randn(1, 2, side, side, generator=torch.Generator().manual_seed(cfg.eval_seed + i)) for i in SAMPLE_PROMPTS])
    calls, trajectory = [], []

    def grab(module, args):
        if len(calls) in SAMPLE_STEPS:
            capture.reset()
        calls.append(args[1][0].item())

    def keep(module, args, out):
        step, x_t, t = len(calls) - 1, args[0].float(), calls[-1]
        n = len(x_t) // 2
        trajectory.append((1 - t) * x_t[:n] + t * (out[n:] + cfg.guidance * (out[:n] - out[n:])))
        if step in SAMPLE_STEPS:
            partials = [("input", x_t, None)] + capture.partials(side // 8, side // 8)
            x0 = []  # CFG on every partial, as the sampler applies it to F
            for label, Fk, _ in partials:
                Fk = Fk[n:] + cfg.guidance * (Fk[:n] - Fk[n:])
                x0.append((label, (1 - t) * x_t[:n] + t * Fk))
            save_strips(decoder, x0, t, step, out_dir)

    hooks = [model.register_forward_pre_hook(grab), model.register_forward_hook(keep)]
    rows = Prompt(*(a[list(SAMPLE_PROMPTS)] for a in prompt))
    euler_sample(model, noise.to(device), rows.to(device, torch.float32), steps, cfg.guidance)
    for hook in hooks:
        hook.remove()
    save_trajectory(decoder, trajectory, out_dir)


def save_trajectory(decoder, trajectory: list, out_dir: Path) -> None:
    """One PNG per prompt: the clean estimate at each sampler step, row-major, 10 steps to a row."""
    images = torch.stack([decoder.decode(z.float()).clamp(-1, 1) for z in trajectory], 1)  # [P, steps, 3, res, res]
    images = ((images + 1) * 127.5).round().byte().cpu().numpy()
    steps, res = images.shape[1], images.shape[-1]
    for p, prompt_index in enumerate(SAMPLE_PROMPTS):
        grid = images[p].reshape(steps // 10, 10, 3, res, res).transpose(0, 3, 1, 4, 2).reshape(steps // 10 * res, 10 * res, 3)
        Image.fromarray(grid).save(out_dir / f"prompt{prompt_index:02d}_trajectory.png")


def save_strips(decoder, x0: list, t: float, step: int, out_dir: Path) -> None:
    """One PNG per prompt: the input's estimate, then one row per attention period, the writes of its blocks in order."""
    labels = [label for label, _ in x0]
    images = torch.stack([decoder.decode(z.float()).clamp(-1, 1) for _, z in x0], 1)  # [P, K, 3, res, res]
    images = ((images + 1) * 127.5).round().byte().cpu().numpy()
    res = images.shape[-1]
    per_row = sum(1 for label in labels if label.startswith(("0.", "1.", "2.", "3.")))
    for p, prompt_index in enumerate(SAMPLE_PROMPTS):
        cells = images[p, 1:]
        rows = len(cells) // per_row
        grid = cells.reshape(rows, per_row, 3, res, res).transpose(0, 3, 1, 4, 2).reshape(rows * res, per_row * res, 3)
        head = np.zeros((res, per_row * res, 3), np.uint8)
        head[:, :res] = images[p, 0].transpose(1, 2, 0)
        Image.fromarray(np.concatenate([head, grid])).save(out_dir / f"prompt{prompt_index:02d}_step{step:02d}_t{t:.3f}.png")
    (out_dir / "columns.txt").write_text("first row: the input x_t's estimate; then per attention period:\n" + " ".join(labels[1 : 1 + per_row]) + "\n")


def main(run_dir: str, step: int, out_dir: str, rows: int = 256) -> None:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16).eval()
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    capture = WriteCapture(model)

    x, prompt = read_rows(cfg.eval_set, cfg.resolution)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))  # the score's noise
    x, prompt, noise = x[:rows], prompt.rows(0, rows), noise[:rows]
    reads = eval_reads(model, capture, x, prompt, noise, t_quantiles(SCORE_TS, SCORE_SHIFT), device)
    (out / "eval_reads.json").write_text(json.dumps(dict(run=run.name, step=step, rows=rows, bands=BANDS, reads=reads)))

    prompts = (run / "eval_prompts.txt").read_text().splitlines()
    sample_reads(model, capture, load_decoder(device), embed_prompts(prompts), cfg, out, device)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], *map(int, sys.argv[4:]))
