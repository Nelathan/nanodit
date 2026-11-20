# The conv block is LFM's gated short conv with the norm on its read

2026-11-20. Closes the conv block's form (PLAN G10, open since archive 2026-11-19-conv-writes-all-channels).

Decided: every conv block is `x += gate_a ⊙ out(C ⊙ rms_norm(conv(B ⊙ v)))`, with B, C and v from one projection of
the modulated input, width → 3 × width, and a depthwise 3×3 conv; it writes all channels and has no branch norm.
Architecture a2 (SPEC §3). The reference for trunk arms is its one-pass run, `a1-lfmnormread-1ep`, which ran on
`arm/lfm-forms` before the merge; its config and last checkpoint are migrated to the trunk's names.

Reasons:
- Over one pass it leads the basic conv at every eval, by 0.0043 at the end, 0.0059 at the same wall-clock, at an
  11.8 % lower throughput (log 2026-11-20-lfm-norm-read-1ep).
- At 6000 steps it led every other form at about the same speed, and the forms judged by the same wall-clock and
  parameters left no cheaper form level with it: GDN2 is slower, sigmoid-gated forms trail by about 0.003, no `out`
  falls behind (log 2026-11-19-conv-forms-gates-and-widths).
- The norm on the read, not after `out`, lets C close the write, so a token can abstain; LFM's form is a product of
  signed linear projections and has no forget term, which is not needed for mixing.

Not run, and left out by the closure: heads sharing a value, one projection per period of the layout.
Measured, not yet acted on: at the end of the pass the coarse band loses R² at t 0.66 to 0.74 against the basic conv.
