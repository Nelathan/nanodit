"""The adaLN modulation of a checkpoint as heatmaps in one self-contained HTML page: per block and t, the RMS of the
write gates (`mod + table` in the gate rows, before any bound) on the slice and on the meta channels; each block's
table rows by channel; how alike the blocks' tables are, the cosine between two blocks' rows about their mean
over blocks, on the slice and on the meta channels; the shared projection's gate rows by t and channel; and the full
gates by block and channel at one t. t runs over 20 evenly spaced values in (0, 1). CPU, fp32: it runs beside a training run.

usage: plot_gates.py RUN_DIR STEP OUT_HTML
"""
import json
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from nanodit.model import GATE_ROWS, N_MOD, TOKEN_DIM, timestep_sinusoid
from nanodit.objective import t_uniform
from train import Config, build_model

TS = 20
ROW_NAMES = ("mixer read scale", "mixer gate", "MLP read scale", "MLP gate")

PAGE = """<!doctype html><meta charset="utf-8"><title>%(title)s</title>
<style>body{font:13px sans-serif;margin:20px}canvas{image-rendering:pixelated;border:1px solid #ccc;margin:2px 0}
.panel{display:inline-block;vertical-align:top;margin:0 24px 24px 0}#tip{position:fixed;background:#fff;border:1px solid #888;padding:2px 5px;pointer-events:none;display:none}</style>
<h2>%(title)s</h2><p>%(note)s</p><div id="root"></div><div id="tip"></div>
<script>
const D = %(data)s;
const tip = document.getElementById("tip");
function color(v, lo, hi, signed) {
  if (signed) { const m = Math.max(-lo, hi) || 1, x = Math.max(-1, Math.min(1, v / m));
    return x >= 0 ? [255, 255 * (1 - x), 255 * (1 - x)] : [255 * (1 + x), 255 * (1 + x), 255]; }
  const x = Math.max(0, Math.min(1, (v - lo) / ((hi - lo) || 1)));
  return [255 * Math.min(1, 2 * x), 255 * Math.max(0, 2 * x - 1) * 0.9 + 20 * (1 - x), 255 * (1 - x) * 0.6];
}
function heat(p) {
  const rows = p.z.length, cols = p.z[0].length, sx = p.sx || Math.max(1, Math.floor(480 / cols)), sy = p.sy || 12;
  const flat = p.z.flat(), lo = p.lo ?? Math.min(...flat), hi = p.hi ?? Math.max(...flat);
  const div = document.createElement("div"); div.className = "panel";
  div.innerHTML = `<b>${p.title}</b><br><small>${p.axes} · range ${lo.toFixed(3)} … ${hi.toFixed(3)}</small><br>`;
  const c = document.createElement("canvas"); c.width = cols * sx; c.height = rows * sy;
  const g = c.getContext("2d");
  for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
    const [R, G, B] = color(p.z[r][k], lo, hi, p.signed); g.fillStyle = `rgb(${R|0},${G|0},${B|0})`; g.fillRect(k * sx, r * sy, sx, sy); }
  if (p.split) { g.fillStyle = "#000"; g.fillRect(p.split * sx, 0, 1, rows * sy); }
  c.onmousemove = e => { const b = c.getBoundingClientRect(), k = Math.floor((e.clientX - b.left - c.clientLeft) / sx), r = Math.floor((e.clientY - b.top - c.clientTop) / sy);
    if (r < 0 || k < 0 || r >= rows || k >= cols) return; tip.style.display = "block"; tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px";
    tip.textContent = `${p.ylabel(r)} · ${p.xlabel(k)} · ${p.z[r][k].toFixed(4)}`; };
  c.onmouseleave = () => tip.style.display = "none";
  div.appendChild(c); return div;
}
const root = document.getElementById("root"), blk = r => `block ${r} (${D.kinds[r]})`, tl = k => `t ${D.t[k].toFixed(3)}`, ch = k => `channel ${k}`;
function section(name) { const h = document.createElement("h3"); h.textContent = name; root.appendChild(h); }
section("Gate RMS by block (rows) and t (columns): mod(t) + table, slice and meta channels");
const rmsHi = Math.max(...D.rms.flat(3));
for (const [b, name] of [[0, "mixer"], [1, "MLP"]]) for (const [l, lane] of [[0, "slice"], [1, "meta"]])
  root.appendChild(heat({title: `${name} gate, ${lane}`, axes: "block × t", z: D.rms.map(blkRow => blkRow.map(tRow => tRow[b][l])), lo: 0, hi: rmsHi, sx: 22, ylabel: blk, xlabel: tl}));
section("The same with the table removed: the shared projection's gate rows alone, by t (one row: the same for every block)");
for (const [b, name] of [[0, "mixer"], [1, "MLP"]]) for (const [l, lane] of [[0, "slice"], [1, "meta"]])
  root.appendChild(heat({title: `${name} shared gate, ${lane}`, axes: "t", z: [D.shared_rms.map(tRow => tRow[b][l])], lo: 0, hi: rmsHi, sx: 22, sy: 20, ylabel: () => "shared", xlabel: tl}));
section("Each block's table by channel (rows: blocks; the black line marks the slice's end)");
D.table.forEach((z, i) => root.appendChild(heat({title: D.row_names[i] + " table", axes: "block × channel", z, signed: true, sx: 1, split: D.token_dim, ylabel: blk, xlabel: ch})));
section("How alike two blocks' tables are: cosine of the rows about their mean over blocks (rows and columns: blocks)");
D.similarity.forEach((lanes, i) => lanes.forEach((z, l) => root.appendChild(heat({title: `${D.row_names[i]}, ${["slice", "meta"][l]}`, axes: "block × block", z, lo: -1, hi: 1, signed: true, sx: 10, sy: 10, ylabel: blk, xlabel: blk}))));
section("The shared projection's gate rows by t and channel");
D.shared.forEach((z, i) => root.appendChild(heat({title: ["mixer", "MLP"][i] + " shared gate", axes: "t × channel", z, signed: true, sx: 1, split: D.token_dim, ylabel: tl, xlabel: ch})));
section(`The full gate, mod + table, by block and channel at t ${D.t_mid.toFixed(3)}`);
D.full_mid.forEach((z, i) => root.appendChild(heat({title: ["mixer", "MLP"][i] + " gate", axes: "block × channel", z, signed: true, sx: 1, split: D.token_dim, ylabel: blk, xlabel: ch})));
</script>
"""


@torch.no_grad()
def main(run_dir: str, step: int, out: str) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    model = build_model(cfg, torch.device("cpu"), torch.float32)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    t = t_uniform(TS)
    mod = model.ada(model.time_mlp(timestep_sinusoid(t))).view(TS, N_MOD, model.width)
    tables = torch.stack([block.table for block in model.blocks])  # [block, N_MOD, width]
    gates = (mod[None] + tables[:, None])[:, :, list(GATE_ROWS)]  # [block, t, 2, width]

    def lanes(g):  # RMS over the slice's channels and over the meta channels: [..., 2]
        return torch.stack([g[..., :TOKEN_DIM].square().mean(-1).sqrt(), g[..., TOKEN_DIM:].square().mean(-1).sqrt()], -1)

    def similarity(rows):  # [block, c] → cosine between blocks of the rows about their mean over blocks
        a = rows - rows.mean(0)
        a = a / a.norm(dim=1, keepdim=True)
        return a @ a.T

    mid = TS // 2
    data = {
        "t": t.tolist(), "t_mid": t[mid].item(), "token_dim": TOKEN_DIM, "row_names": ROW_NAMES,
        "kinds": ["conv" if block.is_conv else "attn" for block in model.blocks],
        "rms": lanes(gates).tolist(),  # [block, t, branch, lane]
        "shared_rms": lanes(mod[:, list(GATE_ROWS)]).tolist(),  # [t, branch, lane]
        "table": tables.transpose(0, 1).tolist(),  # [row, block, channel]
        "shared": mod[:, list(GATE_ROWS)].transpose(0, 1).tolist(),  # [branch, t, channel]
        "full_mid": gates[:, mid].transpose(0, 1).tolist(),  # [branch, block, channel]
        "similarity": [[similarity(tables[:, row, :TOKEN_DIM]).tolist(), similarity(tables[:, row, TOKEN_DIM:]).tolist()] for row in range(N_MOD)],  # [row, lane, block, block]
    }
    note = (f"{run.name} step {step}. Gates before any bound; a write is gate × the branch's output. "
            "Signed maps: red positive, blue negative, white zero, scaled to the largest |value| in the map.")
    Path(out).write_text(PAGE % {"title": f"adaLN modulation, {run.name}", "note": note, "data": json.dumps(data, separators=(",", ":"))})
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
