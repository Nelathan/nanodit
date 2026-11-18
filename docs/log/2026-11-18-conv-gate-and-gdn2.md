# a1-gatedconv-6k and a1-gdn2meta-6k: a gated conv and GDN2's update in the conv blocks

Runs, each `a1-ref-1ep` for 6000 steps with every conv block changed, writing the meta channels only:
- `/mnt/mars/ai/tiny-dit/runs/a1-gatedconv-6k` (branch `arm/gated-conv`, `fd23833`): `σ(W y) ⊙ conv(y)`.
- `/mnt/mars/ai/tiny-dit/runs/a1-gdn2meta-6k` (branch `arm/gdn2-meta`, `95fe032`): `σ(write) ⊙ silu(conv(value)) −
  σ(forget) ⊙ state`, the four from one projection 768 → 3072 of the block's input (the conv block of `12c2028`'s parent).

Against `a1-ref-1ep` (seed 41) and `a1-seed42-6k` at the same step; the two references differ by 0.0009 at 6000
(log 2026-11-18-seed42), so a gap under about 0.002 is not resolved.

| Step | gated | Δ ref | Δ seed 42 | GDN2 | Δ ref | Δ seed 42 |
|---|---|---|---|---|---|---|
| 1000 | 1.5553 | −0.0009 | −0.0002 | 1.5512 | −0.0049 | −0.0043 |
| 2000 | 1.5144 | −0.0041 | −0.0028 | 1.5130 | −0.0054 | −0.0041 |
| 3000 | 1.4956 | −0.0025 | −0.0016 | 1.4930 | −0.0052 | −0.0042 |
| 4000 | 1.4829 | −0.0028 | −0.0018 | 1.4807 | −0.0050 | −0.0040 |
| 5000 | 1.4736 | −0.0034 | −0.0023 | 1.4709 | −0.0061 | −0.0050 |
| 6000 | 1.4675 | −0.0019 | −0.0009 | 1.4641 | −0.0053 | −0.0044 |

Δ against `a1-ref-1ep` per t at 6000, the score's 10 t (0.16 … 0.84):

| Arm | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| gated | −0.0010 | −0.0028 | −0.0037 | −0.0036 | −0.0030 | −0.0026 | −0.0023 | −0.0006 | −0.0000 | +0.0006 |
| GDN2 | −0.0042 | −0.0069 | −0.0076 | −0.0070 | −0.0062 | −0.0057 | −0.0058 | −0.0052 | −0.0044 | −0.0001 |

Held-out rows at 6000: gated 1.3106, GDN2 1.3071 (the references do not score them). GDN2's band R² at 6000 (over
three tokens, one to three, inside one): Lunara 0.809, 0.709, 0.615; held-out 0.861, 0.743, 0.640. The gated arm and
the references predate the band score.

Throughput, median `imgs_per_s` to step 6000 after the first five records: reference 593, seed 42 599, gated 559
(0.94×), GDN2 505 (0.85×). GDN2 adds 18 × 768 × 3072 = 42.5M parameters.

## Read

GDN2 is ahead at every eval by 0.004 to 0.005 against seed 42 and 0.005 to 0.006 against seed 41, twice the seed
draw or more, and at every t but the noisiest; at depth 12 its lead was 0.0014 (FACTS, Architecture at d768x12). The
output gate alone is within the seed noise at 6000. What carries GDN2's lead is among the parts the gated arm lacks:
the value projection before the conv, the silu on the read, and the forget term on a projected state.
