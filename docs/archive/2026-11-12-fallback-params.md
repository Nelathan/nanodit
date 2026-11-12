# Closed: how Usui treats 1-D parameters

2026-11-12. Decided, by reading the code. Usui rejects any parameter that is not a matrix (`add_param_group` in
usuitrack's `optimizer.py`), so norm gains, biases, the null embedding and the adaLN tables train on the caller's
fallback, `StochasticAdamW` (SPEC §6). The split is `src/nanodit/optim.py`, and its test routes every parameter
exactly once.
