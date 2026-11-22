# a4 at 6000 steps: the adaLN tables on the held meta stream with long skips

Runs (each `config.toml` in its run directory; a4 on the arm's code, seed 41, 6000 steps), against
`a4-held-meta-skips-6k` (`arm/held-meta-skips` `d178c97`, log 2026-11-22-stream-arms) and `a4-base-6k` (main):

| Run | Code | Change on held meta + long skips |
|---|---|---|
| `a4-held-meta-skips-no-tables-6k` | `arm/held-meta-skips-no-tables` `3db47ca` | the per-block adaLN tables removed: every block gets the shared projection's read scales and write gates, as on `arm/no-tables` |
| `a4-held-meta-skips-scalar-gates-6k` | `arm/held-meta-skips-scalar-gates` `dcfc8e7` | the tables and the shared gate rows removed: the read scales from the shared projection alone, each write gated by one per-block scalar through a tanh, from zero and independent of t; the MLP's output norm without a gain |

Both score the uniform grid natively. The seed spread at 6000 steps is about 0.001; a difference under about 0.002 is
not resolved (log 2026-11-18-seed42).

## Score at 6000

| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| held meta + long skips | 1.2999 | −0.0071 | 1.3544 | −0.0065 | 1.2520 | −0.0051 | 1.2922 | −0.0050 |
| no tables | 1.3175 | +0.0105 | 1.3709 | +0.0100 | 1.2654 | +0.0083 | 1.3050 | +0.0079 |
| scalar gates | 1.3244 | +0.0174 | 1.3753 | +0.0145 | 1.2704 | +0.0133 | 1.3078 | +0.0107 |

Δ against the base. Against held meta + long skips: no tables +0.0176 / +0.0165 / +0.0134 / +0.0128, scalar gates
+0.0245 / +0.0209 / +0.0184 / +0.0157. Each Δ is from the unrounded scores.

Uniform grid, held-out: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1171 | 1.2121 | 1.2582 | 1.2857 | 1.3129 | 1.3448 | 1.3705 | 1.3516 | 1.2416 | 1.0767 |
| held meta + long skips | +0.0019 | −0.0026 | −0.0052 | −0.0076 | −0.0091 | −0.0096 | −0.0086 | −0.0058 | −0.0033 | −0.0013 |
| no tables | +0.0039 | +0.0097 | +0.0164 | +0.0197 | +0.0177 | +0.0115 | +0.0039 | −0.0007 | −0.0007 | +0.0017 |
| scalar gates | +0.0060 | +0.0129 | +0.0227 | +0.0279 | +0.0267 | +0.0206 | +0.0122 | +0.0040 | +0.0006 | −0.0010 |

Uniform grid, Lunara: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1214 | 1.2505 | 1.3206 | 1.3549 | 1.3782 | 1.4031 | 1.4226 | 1.3936 | 1.2561 | 1.0699 |
| held meta + long skips | +0.0006 | −0.0028 | −0.0050 | −0.0074 | −0.0088 | −0.0090 | −0.0068 | −0.0043 | −0.0027 | −0.0034 |
| no tables | +0.0019 | +0.0073 | +0.0148 | +0.0185 | +0.0164 | +0.0105 | +0.0048 | +0.0019 | +0.0011 | +0.0013 |
| scalar gates | +0.0032 | +0.0099 | +0.0204 | +0.0261 | +0.0244 | +0.0177 | +0.0085 | +0.0001 | −0.0017 | −0.0015 |

## Read

- **The held stream does not make the tables free.** Removing them costs +0.013 to +0.018 against held meta + long
  skips; on the old stream it cost +0.011 to +0.015 against the base (log 2026-11-22-stream-arms). The cost sits in the
  same place, mid-t, peaking at t 0.35. The topic's reading of the tables as a source of the gates' magnitude, a role
  the held stream removes, does not account for what they carry.
- **One scalar per write loses more than the shared gates.** Scalar gates is behind no tables by 0.003 to 0.007 on
  the four scores, the smallest gap (Lunara uniform) barely resolved, with the same mid-t peak at t 0.35, larger. It
  differs from no tables in three things at once: no gate depends on t, no gate varies by channel, and the MLP's output
  norm loses its gain, so a per-channel profile of the MLP's write is no longer absorbed by `w_down`; this run does not
  separate them.
- **What the tables carry is each block's own baseline per channel** (*Hypothesis*). The t-dependence is the shared
  projection's in every block; a table adds a constant. On a read, (1 + s(t) + table) is (1 + table) times
  (1 + s(t) / (1 + table)): the block's linear maps absorb the factor, not the changed depth of the t-modulation. On a
  gate, g(t) + table gives the write a baseline the shared gate rows, starting at zero and the same in every block, do
  not. Without the tables every block has the same modulation depth per channel and no gate baseline. Which of the read
  scales and the gates the cost sits in is not separated.
