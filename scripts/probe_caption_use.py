"""How much a run's checkpoint uses the caption, and where its attention reads it. Over the first ROWS rows of the eval
set at the score's 10 t and its noise:

- `loss`: the loss with each row's caption (`caption`) and with the null prompt in its place (`null`).
- per attention block, the RMS of its attention writes on the stream, over all channels and over the image channels,
  each a mean over rows: the write gate_a ⊙ out(·) and its split into the part read from the image keys and the part
  read from the caption keys. The split is exact: each head's read is scaled by one number per
  query (its RMS norm, the sink) and then by the Krea gate, `out` is linear without a bias, so the write is the sum of
  the two parts taken through the same scaling. `split_error`: the recombined write's RMS distance from the block's,
  relative, a mean over rows.
- per attention block and head, each a mean over queries: `caption`, the softmax mass on the caption's keys (the
  real keys only; the null key is not among them, as it is not in lse), against `caption_share`, the caption's share
  of the real keys, L / (tokens + L), what uniform attention would give; `open`: σ(lse − b), the share
  of the head's read the null key leaves, `closed`: the fraction of queries with open < 0.1, and `caption_open`:
  open × caption. Each also by the quartile of the query token's detail: the RMS of its clean latent patch about the
  patch mean, a proxy for flat background (lowest) against texture and edges (highest); the quartiles are of the clean
  latent, and at high t the model sees mostly noise there.

The model runs uncompiled; a wrapper around `model.joint_attention` takes each call's q, keys and values, recomputes
the logits explicitly (64 + L keys at 128 px) and checks its lse against flex's.

usage: probe_caption_use.py RUN_DIR STEP OUT_JSON [ROWS, default 256]
"""
import json
from collections import defaultdict
import sys
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import nanodit.model as nanodit_model
from nanodit.data import read_rows
from nanodit.model import GATE_ROWS, HEAD_DIM, N_MOD, TOKEN_DIM, patchify
from nanodit.objective import noisy, per_sample_loss, t_quantiles
from train import SCORE_SHIFT, SCORE_TS, Config, build_model

BATCH = 16
QUARTILES = 4


class Mean:
    """A running sum of per-row values (any trailing shape) and of per-token values by detail quartile."""

    def __init__(self):
        self.rows, self.by_detail = 0.0, 0.0

    def add_rows(self, values):  # [B, ...]
        self.rows = self.rows + values.sum(0)

    def add_tokens(self, values, onehot):  # values [B, H, N], onehot [B, N, Q]
        self.add_rows(values.mean(-1))
        self.by_detail = self.by_detail + torch.einsum("bhn,bnq->qh", values, onehot)


class Capture:
    """Per forward, while `on`: the attention writes and the per-head reads of every attention block."""

    def __init__(self, model):
        self.model, self.on, self.block, self.mod, self.records = model, False, None, None, {}
        self.attention = nanodit_model.joint_attention
        nanodit_model.joint_attention = self.joint_attention
        model.ada.register_forward_hook(lambda m, i, out: setattr(self, "mod", out.view(out.shape[0], N_MOD, -1)))
        for b, block in enumerate(model.blocks):
            if block.is_conv:
                continue
            block.register_forward_pre_hook(lambda m, args, b=b: setattr(self, "block", b))
            block.q_norm.register_forward_hook(lambda m, i, out, b=b: self.null_logit(b, out))
            block.gate.register_forward_hook(lambda m, i, out, b=b: self.record(b, gate=torch.sigmoid(out.float())))
            block.out.register_forward_hook(lambda m, i, out, b=b: self.joint_write(b, out))

    def gate_a(self, b):
        return (self.mod + self.model.blocks[b].table)[:, GATE_ROWS[0], None]

    def record(self, b, **values):
        if self.on:
            self.records.setdefault(b, {}).update(values)

    def null_logit(self, b, q_normed):
        """b of `Block.joint_read`: the pre-RoPE normed query against the null key."""
        block = self.model.blocks[b]
        logit = (q_normed.float().unflatten(1, (block.kv_heads, -1)) @ block.null_key[:, None, :, None].float()).flatten(1, 2).squeeze(-1)
        self.record(b, null_logit=logit * HEAD_DIM**-0.5)

    def joint_attention(self, q, k, v, mask):
        out, lse = self.attention(q, k, v, mask)
        if self.on:
            N, group = q.shape[2], q.shape[1] // k.shape[1]
            keep = torch.cat([mask.new_ones(mask.shape[0], N), mask], dim=1)
            scores = (q.float() @ k.float().repeat_interleave(group, 1).transpose(-1, -2)) * HEAD_DIM**-0.5
            scores = scores.masked_fill(~keep[:, None, None], -float("inf"))
            assert (scores.logsumexp(-1) - lse.float()).abs().max() < 1e-2, "the explicit logits do not give flex's lse"
            weights = scores.softmax(-1)
            caption_read = weights[..., N:] @ v[:, :, N:].float().repeat_interleave(group, 1)  # [B, H, N, D]
            self.record(self.block, caption=weights[..., N:].sum(-1), lse=lse.float(), read=out, caption_read=caption_read)
        return out, lse

    def joint_write(self, b, out):
        """The write and its caption part, taken through `Block.joint_read`'s scaling, the Krea gate and `out`."""
        if not self.on:
            return
        record = self.records[b]
        read = record["read"].float()
        scale = torch.rsqrt(read.square().mean(-1, keepdim=True) + torch.finfo(record["read"].dtype).eps)  # F.rms_norm's
        record["open"] = torch.sigmoid(record["lse"] - record["null_logit"])
        scale = scale * record["open"][..., None]
        write, caption_write, recombined = self.gate_a(b) * out, self.to_stream(b, record["caption_read"], scale), self.to_stream(b, read, scale)
        # the block runs the chain in bf16, this split in fp32: 2 % apart in RMS at init; recorded, not asserted
        split_error = (recombined - write.float()).square().mean((1, 2)).sqrt() / write.float().square().mean((1, 2)).sqrt()
        record.update(write=write, caption_write=caption_write, image_write=recombined - caption_write, split_error=split_error)

    def to_stream(self, b, read, scale):
        """A per-head read [B, H, N, D] scaled per query and head, then the Krea gate, `out` and gate_a, in fp32."""
        block, B, N = self.model.blocks[b], read.shape[0], read.shape[2]
        gated = (read * scale).transpose(1, 2).reshape(B, N, -1) * self.records[b]["gate"]
        return self.gate_a(b).float() * F.linear(gated, block.out.weight.float())


CONDITIONS = ("caption", "null")
WRITES = ("write", "image_write", "caption_write")


def row_rms(write):
    """Per row, the RMS over tokens and channels, over all channels and over the image channels: [B, 2]."""
    return torch.stack([write.float().square().mean((1, 2)).sqrt(), write[..., :TOKEN_DIM].float().square().mean((1, 2)).sqrt()], -1)


@torch.no_grad()
def main(run_dir: str, step: int, out_json: str, rows: int = 256) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16).eval()
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    capture = Capture(model)
    x, prompt = read_rows(cfg.eval_set, cfg.resolution)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))
    x, prompt, noise = x[:rows], prompt.rows(0, rows), noise[:rows]
    detail = patchify(x.float())[0].std(-1)  # [rows, tokens]: RMS of each clean patch about its mean
    edges = detail.flatten().quantile(torch.linspace(0, 1, QUARTILES + 1)[1:-1])
    quartile = torch.bucketize(detail, edges).to(device)  # [rows, tokens] in 0 .. QUARTILES - 1
    attention = [b for b, block in enumerate(model.blocks) if not block.is_conv]

    reads = []
    for t_value in t_quantiles(SCORE_TS, SCORE_SHIFT).tolist():
        losses = {name: Mean() for name in CONDITIONS}
        sums = {b: defaultdict(Mean) for b in attention}
        share, count_q = Mean(), torch.zeros(QUARTILES, device=device)
        for i in range(0, len(x), BATCH):
            xb, eb = x[i : i + BATCH].to(device, torch.float32), noise[i : i + BATCH].to(device)
            t = torch.full((len(xb),), t_value, device=device)
            x_t, rows_prompt = noisy(xb, eb, t), prompt.rows(i, i + BATCH).to(device, torch.float32)
            null = model.null_prompt(*rows_prompt.tokens.shape[:2])
            for name, condition in zip(CONDITIONS, (rows_prompt, null)):
                capture.on, capture.records = name == "caption", {}
                losses[name].add_rows(per_sample_loss(model(x_t, t, condition), xb, eb, x_t))
                if name == "caption":
                    records = capture.records
            capture.on = False
            length = rows_prompt.mask.sum(1).float()
            share.add_rows(length / (detail.shape[1] + length))
            onehot = F.one_hot(quartile[i : i + BATCH], QUARTILES).float()  # [B, N, Q]
            count_q += onehot.sum((0, 1))
            for b, record in records.items():
                for name in WRITES:
                    sums[b][name].add_rows(row_rms(record[name]))
                sums[b]["split_error"].add_rows(record["split_error"])
                sums[b]["caption"].add_tokens(record["caption"], onehot)  # each [B, H, N]
                open_ = record["open"]
                for m, v in dict(open=open_, closed=(open_ < 0.1).float(), caption_open=open_ * record["caption"]).items():
                    sums[b][m].add_tokens(v, onehot)
        n = len(x)
        blocks = []
        for b in attention:
            entry = dict(block=b)
            for name, mean in sums[b].items():
                if name in WRITES:
                    entry[name] = dict(zip(("all", "image"), (mean.rows / n).tolist()))
                elif name == "split_error":
                    entry[name] = (mean.rows / n).item()
                else:
                    entry |= {name: (mean.rows / n).tolist(), name + "_by_detail": (mean.by_detail / count_q[:, None]).tolist()}
            blocks.append(entry)
        reads.append(dict(t=t_value, loss={k: (v.rows / n).item() for k, v in losses.items()}, caption_share=(share.rows / n).item(), blocks=blocks))
    Path(out_json).write_text(json.dumps(dict(run=run.name, step=step, rows=rows, quartiles=QUARTILES, reads=reads)))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], *map(int, sys.argv[4:]))
