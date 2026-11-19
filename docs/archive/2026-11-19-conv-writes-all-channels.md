# The conv block writes all channels; its form must gate well enough to do so

2026-11-19. Reverses the meta-only write of the conv block (PLAN G10, decided 2026-11-16).

Decided: a conv block writes all channels by default, the 128 image channels included. A conv form is acceptable only
if it is gated well enough to write them safely; the meta-only write was what the basic conv needed, not a property
of the design. SPEC keeps the basic conv writing the meta channels only until a gated form is chosen, since the basic
conv is the form that cannot write all channels.

Reasons (log 2026-11-19-conv-forms-and-attn-middle-of-three): writing all channels costs no step time and gains 0.0013
(LFM's gated short conv) to 0.0019 (GDN2's update) at 6000 steps in both gated forms. A gate from the token's own
state, per channel, decides what it admits (*Hypothesis*: so the image channels are not washed; no smearing
is seen in the LFM samples). Writing only the meta channels
is easier to learn; a lead at the first evals is an observation, not a verdict.

How a conv form is judged from here, beside the score at the same step: the score at the same wall-clock, and its
parameter count, since parameters and step time could buy more blocks instead. A form that needs a stronger MLP to
make up for its mixing is not wanted; the conv carries its own mixing.
