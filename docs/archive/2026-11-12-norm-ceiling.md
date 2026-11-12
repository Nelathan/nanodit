# A norm ceiling in place of the pin

2026-11-12. Closes the norm-bounds item of PLAN G8.

Decided: every Usui matrix's Frobenius norm is clipped from above at `norm_max` times its init norm after each step,
with no floor and no weight decay (SPEC §6). Usui carries this as `norm_bounds = (low, high)` per param group, in
multiples of the norm before the first write, either side open; the pin is the special case (1, 1) and the same code.

Reasons: training grows weights, so the bound that matters is the upper one; a floor only holds a norm that a step
shrinks, which a decay or a pin would force and nothing here asks for. A clamp is one radial rescale inside the
write that already happens, so it costs no more code or time than the pin. Decay as a pull toward a norm would add
a second rate, tied to the lr, and is not wanted until an arm needs it.

Not decided: the ceiling's height. The reference config sets 1.0, which keeps it comparable with the pinned arms;
the height is an arm in PLAN G8, read with the weight norm and stable rank logged at every eval (SPEC §8).
