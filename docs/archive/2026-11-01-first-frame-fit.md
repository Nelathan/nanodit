# Usui's first frame fit, 2026-11-01

Closes PLAN A3. Decided by Daniel after the GPU failure was isolated (FACTS, First frame fit on the GPU). Each entry: what, and why.

| # | Decision | Reason |
|---|---|---|
| 1 | Usui fits a matrix's first frame with a random range finder, one power step, from a private generator with a fixed seed, so the training random stream never moves; no `eigh` at the fit (usuitrack `zero-init` `e7e5210`, its SPEC step 2) | cuSOLVER's fp32 `eigh` fails on a nearly rank-one Gram, and the failure is fragile (scaling by 3 or 1e-6 noise make it pass). Daniel: "jidder is bad for accuracy. But it fails because its degenerate"; "Its a special case where actually a random init is just as good. Fp64 for init is ok but seem indulgent. For gram ismts wadteful on the hot path"; "random range finder please because robust". Against the exact eigenspace, the range finder captures 0.99981 of the failing Gram's trace against 1.0, and at least 99.27% of the exact capture on all 78 first fits of the depth-12 model at micro-batch 64 (usuitrack `docs/REVIEW_zero-init.md` at `619b6e8`) |
