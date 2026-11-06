"""The gradient-moment cosine of a noisy quadratic under EMA momentum, against its distance to bubbleforce (PLAN, edge of stability).

usage: probe_momentum_cosine.py

UsuiTrack's moment is `M = beta M + (1 - beta) g` and the write is `-lr * M` (before the polar map), the momentum kernel
`c_j = (1 - beta) beta^j` of the bubbleforce post. On one quadratic direction of sharpness `lam` with white gradient noise,
the characteristic polynomial is `z^2 - (1 + beta - eta lam (1 - beta)) z + beta`; its roots reach -1 at
`eta lam = 2 (1 + beta) / (1 - beta)`. Prints, per beta and per `eta lam` as a fraction of that, the root radius and the
stationary `cos(g_t, M_{t-1})`, the quantity of the `grad_moment_cosine` diagnostic. Then fits the fraction at which
beta 0.9 reads the real run's -0.41 (FACTS, `128_fixed_2e-3_9k`) and predicts beta 0.99 at the same sharpness and lr.
"""
import numpy as np

STEPS = 200_000
NOISE = 0.01  # gradient noise std; the cosine depends on eta*lam/bubbleforce, not on this scale


def stationary_cosine(beta, fraction, rng):
    eta = fraction * bubbleforce(beta)  # lam = 1
    w, m = 0.0, 0.0
    gm = gg = mm = 0.0
    for t in range(STEPS):
        g = w + NOISE * rng.standard_normal()
        if t > STEPS // 4:
            gm += g * m
            gg += g * g
            mm += m * m
        m = beta * m + (1 - beta) * g
        w -= eta * m
    radius = np.abs(np.roots([1, -(1 + beta - eta * (1 - beta)), beta])).max()
    return gm / np.sqrt(gg * mm), radius


def fitted_fraction(beta, target, rng):
    """The fraction of bubbleforce at which the cosine reads `target`, by bisection (the cosine falls as the fraction rises)."""
    low, high = 0.01, 0.99
    for _ in range(10):
        mid = (low + high) / 2
        if stationary_cosine(beta, mid, rng)[0] > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def bubbleforce(beta):
    return 2 * (1 + beta) / (1 - beta)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for beta in (0.9, 0.95, 0.99):
        print(f"beta {beta}: bubbleforce {bubbleforce(beta):.0f}")
        for fraction in (0.01, 0.1, 0.3, 0.6, 0.9, 0.99):
            cosine, radius = stationary_cosine(beta, fraction, rng)
            print(f"  eta*lam / bubbleforce {fraction:4.2f}: cosine {cosine:+.3f}, root radius {radius:.4f}")
    fraction = fitted_fraction(0.9, -0.41, rng)
    scaled = fraction * bubbleforce(0.9) / bubbleforce(0.99)
    print(f"beta 0.9 reads -0.41 at {fraction:.3f} of bubbleforce; the same eta*lam at beta 0.99 is {scaled:.4f} of it: cosine {stationary_cosine(0.99, scaled, rng)[0]:+.3f}")
