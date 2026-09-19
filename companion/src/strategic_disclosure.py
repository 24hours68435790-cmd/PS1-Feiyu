"""
strategic_disclosure.py
=======================
Reference implementation for

    "When the Model Becomes the Target: Strategic Disclosure under
     AI-Assisted Valuation"  (COMSCI/ECON 206, PS1, Feiyu Li)

The module does two separable jobs.

(1) REPLICATION.  `Space` re-implements, in Python, the exact arithmetic of the
    deployed Hugging Face Static Space (index.html, commit e6a0f5f), including
    JavaScript's `mulberry32` PRNG and `Math.imul`.  Same seed, same draw, same
    epsilon sequence, same prices.  This is what makes the browser demo and the
    notebook the *same* model rather than two things that merely look alike.

(2) EXPERIMENT.  `sweep_*` functions run the comparisons the browser cannot:
    outcome-noise levels, evaluator memory strength lambda, the full type space,
    and -- following the instructor's Week 2 note -- a *committed* (announced =
    followed) rule against a *revisable* (announced allowance, followed rule
    updated from this firm's own history) rule, recording DEVIATION GAIN as well
    as disclosure accuracy.

Nothing here touches real firm data.  Every quantity is simulated.

Author: Feiyu Li.  Licence: MIT (see LICENSE).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Model constants -- identical to the deployed index.html
# --------------------------------------------------------------------------
M_BAR = 1.0        # the static evaluator's fixed, announced inflation allowance
LAMBDA = 1.0       # how hard the revisable evaluator discounts last round's residual
EFFORT_COST = 0.5  # cost of productive effort a = 1

THETAS = (1, 2, 3)          # private quality
KAPPAS = (0.5, 1.0, 2.0)    # private manipulation-cost parameter k
EFFORTS = (0, 1)            # productive effort a
WD_GRID = (0.0, 1.0, 2.0)   # window-dressing m, browser grid


# ==========================================================================
# 1.  Bit-exact replication of the deployed Space
# ==========================================================================
def _imul(a: int, b: int) -> int:
    """JavaScript Math.imul, kept as an unsigned 32-bit bit pattern."""
    return (a * b) & 0xFFFFFFFF


def mulberry32(seed: int):
    """Exact port of the `mulberry32` closure in index.html."""
    t = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal t
        t = (t + 0x6D2B79F5) & 0xFFFFFFFF
        x = _imul(t ^ (t >> 15), (1 | t) & 0xFFFFFFFF)
        x = ((x + _imul(x ^ (x >> 7), (61 | x) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ x
        x &= 0xFFFFFFFF
        return ((x ^ (x >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


@dataclass
class RoundRecord:
    round: int
    a: float
    m: float
    v: float
    s: float
    eps: float
    y: float
    p_static: float
    p_memory: float
    p: float
    mispricing: float
    payoff: float
    welfare: float
    g: float


@dataclass
class Space:
    """Python twin of the deployed browser artifact.

    `sigma` and `lam` mirror the two controls the page exposes. Their defaults
    (0.5, 1.0) are the page's defaults and reproduce the Week 2 version
    bit-for-bit, which is what `verify_space_replication()` asserts.

    >>> sp = Space(seed=2026)
    >>> sp.theta, sp.k
    (2, 0.5)
    """

    seed: int = 2026
    mode: str = "memory"           # "memory" | "static"
    sigma: float = 0.5             # outcome-noise control; eps in {-sigma, 0, +sigma}
    lam: float = LAMBDA            # memory-strength control
    theta: int = field(init=False)
    k: float = field(init=False)
    history: list = field(default_factory=list, init=False)
    log: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rng = mulberry32(self.seed)
        self.theta = THETAS[int(self._rng() * 3)]
        self.k = KAPPAS[int(self._rng() * 3)]

    def submit(self, a: float, m: float) -> RoundRecord:
        """One round, transcribed from `submit()` in index.html."""
        if not (0 <= m <= 2):
            raise ValueError(
                f"{m} is outside the valid range 0-2; the Space rejects the "
                "report, does not advance the round, and modifies no state."
            )
        rnd = len(self.history) + 1
        v = self.theta + a
        s = self.theta + a + m
        eps = (-self.sigma, 0.0, self.sigma)[int(self._rng() * 3)]
        y = v + eps

        p_static = max(0.0, s - M_BAR)
        if not self.history:
            p_memory = p_static
        else:
            g1 = self.history[0].g
            p_memory = max(0.0, s - M_BAR - self.lam * g1)

        p = p_memory if self.mode == "memory" else p_static
        manip = self.k * m * m
        rec = RoundRecord(
            round=rnd, a=a, m=m, v=v, s=s, eps=eps, y=y,
            p_static=p_static, p_memory=p_memory, p=p,
            mispricing=abs(p - v),
            payoff=p - EFFORT_COST * a - manip,
            welfare=v - manip,
            g=(s - y) - M_BAR,
        )
        self.history.append(rec)
        self.log.append(rec)
        return rec


# ==========================================================================
# 2.  The two evaluator rules, stated as the instructor asked
# ==========================================================================
# COMMITTED  : the announced rule IS the followed rule.  Allowance m_bar in
#              every round, for every firm, whatever the history says.
# REVISABLE  : the announced allowance is still m_bar, but after round 1 the
#              evaluator revises the rule it will actually follow using this
#              firm's own residual g_1.  Announced != followed -- the Banchio,
#              Skrzypacz & Yang (2025) distinction, transplanted from auction
#              design to valuation.
# --------------------------------------------------------------------------
def price_committed(s: float) -> float:
    return max(0.0, s - M_BAR)


def price_revisable(s: float, g1: float, lam: float = LAMBDA) -> float:
    return max(0.0, s - M_BAR - lam * g1)


def residual(s1: float, y1: float) -> float:
    """g_1: the inflation the evaluator can infer from round 1."""
    return (s1 - y1) - M_BAR


# ==========================================================================
# 3.  Two-period play under a general noise level
# ==========================================================================
def play_two_periods(theta, k, a1, m1, a2, m2, eps1, eps2, rule, lam=LAMBDA):
    """Return (total_payoff, total_welfare, mispricing_1, mispricing_2)."""
    v1, s1 = theta + a1, theta + a1 + m1
    v2, s2 = theta + a2, theta + a2 + m2
    y1 = v1 + eps1

    p1 = price_committed(s1)                       # round 1 is identical either way
    if rule == "committed":
        p2 = price_committed(s2)
    else:
        p2 = price_revisable(s2, residual(s1, y1), lam)

    pay = (p1 - EFFORT_COST * a1 - k * m1 ** 2) + (p2 - EFFORT_COST * a2 - k * m2 ** 2)
    wel = (v1 - k * m1 ** 2) + (v2 - k * m2 ** 2)
    return pay, wel, abs(p1 - v1), abs(p2 - v2)


def _eps_draws(sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Outcome noise. sigma = 0 reproduces the browser's degenerate case."""
    if sigma == 0:
        return np.zeros(n)
    return rng.uniform(-sigma, sigma, size=n)


def best_response(theta, k, rule, sigma, lam=LAMBDA, n_draws=20000, seed=0,
                  wd_grid=WD_GRID):
    """Expected-payoff-maximising (a1, m1, a2, m2) for one firm type.

    Returns a dict with the chosen strategy and the outcome families the
    proposal reports, including DEVIATION GAIN: the expected payoff of the best
    strategy minus the expected payoff of reporting honestly (m1 = m2 = 0) while
    keeping the same effort.  A deviation gain of 0 means the rule leaves a
    truthful firm nothing to gain by inflating.

    Only eps_1 enters any payoff -- round 2's outcome is realised after the last
    price is posted -- so the expectation is taken over eps_1 alone.  The scalar
    loop in `play_two_periods` is the readable definition; this is its vectorised
    twin, and `verify_vectorisation()` checks they agree.
    """
    rng = np.random.default_rng(seed)
    e1 = _eps_draws(sigma, n_draws, rng)

    def expected(a1, m1, a2, m2):
        v1, s1 = theta + a1, theta + a1 + m1
        v2, s2 = theta + a2, theta + a2 + m2
        p1 = price_committed(s1)
        if rule == "committed":
            p2 = np.full_like(e1, price_committed(s2))
        else:
            g1 = (s1 - (v1 + e1)) - M_BAR
            p2 = np.maximum(0.0, s2 - M_BAR - lam * g1)
        pay = (p1 - EFFORT_COST * a1 - k * m1 ** 2) + (p2 - EFFORT_COST * a2 - k * m2 ** 2)
        wel = (v1 - k * m1 ** 2) + (v2 - k * m2 ** 2)
        mis = (abs(p1 - v1) + np.abs(p2 - v2)) / 2.0
        return float(pay.mean()), float(wel), float(mis.mean())

    grid = list(itertools.product(EFFORTS, wd_grid, EFFORTS, wd_grid))
    scored = {g: expected(*g) for g in grid}
    star = max(scored, key=lambda g: scored[g][0])
    a1, m1, a2, m2 = star
    pay, wel, mis = scored[star]

    honest_pay = scored[(a1, 0.0, a2, 0.0)][0]      # same effort, zero inflation
    return {
        "theta": theta, "k": k, "rule": rule, "sigma": sigma, "lambda": lam,
        "a1": a1, "m1": m1, "a2": a2, "m2": m2,
        "payoff": pay, "welfare": wel, "mispricing": mis,
        "manipulation": (m1 + m2) / 2,
        "honest": float(m1 == 0 and m2 == 0),
        "deviation_gain": pay - honest_pay,
    }


# ==========================================================================
# 4.  The measurement the peer reviewers asked for
# ==========================================================================
def false_punishment_rate(sigma: float, lam: float = LAMBDA, n=200_000, seed=7):
    """P(a revisable evaluator prices an HONEST firm below its true value)
    and the expected size of that error, versus the committed rule.

    An honest firm sets m = 0, so s = v, y = v + eps, g_1 = -eps - m_bar, and
    the revisable round-2 price is max(0, v - m_bar + lam*(eps + m_bar)).
    """
    rng = np.random.default_rng(seed)
    eps = _eps_draws(sigma, n, rng)
    v = 3.0                                      # theta = 2, a = 1; level cancels
    p_rev = np.maximum(0.0, v - M_BAR + lam * (eps + M_BAR))
    p_com = np.maximum(0.0, v - M_BAR)
    return {
        "sigma": sigma, "lambda": lam,
        "under_priced_share": float(np.mean(p_rev < v)),
        "worse_than_committed_share": float(np.mean(np.abs(p_rev - v) > np.abs(p_com - v))),
        "mae_revisable": float(np.mean(np.abs(p_rev - v))),
        "mae_committed": float(np.mean(np.abs(p_com - v))),
    }


def verify_vectorisation(tol=1e-9) -> bool:
    """The readable scalar definition and the vectorised expectation must agree."""
    rng = np.random.default_rng(123)
    e1 = rng.uniform(-0.75, 0.75, size=3000)
    theta, k, a1, m1, a2, m2, lam = 2, 1.0, 1, 2.0, 0, 1.0, 1.3
    for rule in ("committed", "revisable"):
        scalar = np.mean([
            play_two_periods(theta, k, a1, m1, a2, m2, x, 0.0, rule, lam)[0] for x in e1
        ])
        v1, s1 = theta + a1, theta + a1 + m1
        s2 = theta + a2 + m2
        p1 = price_committed(s1)
        if rule == "committed":
            p2 = np.full_like(e1, price_committed(s2))
        else:
            p2 = np.maximum(0.0, s2 - M_BAR - lam * ((s1 - (v1 + e1)) - M_BAR))
        vec = float(np.mean((p1 - EFFORT_COST * a1 - k * m1 ** 2)
                            + (p2 - EFFORT_COST * a2 - k * m2 ** 2)))
        if abs(scalar - vec) > tol:
            raise AssertionError(f"{rule}: scalar {scalar} != vectorised {vec}")
    return True


def verify_space_replication() -> bool:
    """The Python twin must reproduce the four cases logged from the browser."""
    sp = Space(seed=2026, mode="memory")
    assert (sp.theta, sp.k) == (2, 0.5), "seed 2026 must draw theta=2, k=0.5"
    r1, r2 = sp.submit(0, 2), sp.submit(0, 2)
    assert (r1.p, r1.mispricing, r1.payoff, r1.g) == (3.0, 1.0, 1.0, 1.0)
    assert (r2.p, r2.mispricing, r2.payoff) == (2.0, 0.0, 0.0)

    st = Space(seed=2026, mode="static")
    assert [st.submit(0, 2).p for _ in range(2)] == [3.0, 3.0]

    hm = Space(seed=2026, mode="memory")
    assert [hm.submit(1, 0).p for _ in range(2)] == [2.0, 3.0]

    hs = Space(seed=2026, mode="static")
    assert [hs.submit(1, 0).p for _ in range(2)] == [2.0, 2.0]

    try:
        Space(seed=2026).submit(0, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("m = 7 must be rejected, not clamped")

    # The sigma/lambda controls added for v2 must be strictly additive: at the
    # page's defaults nothing may move (asserted above). Beyond the defaults the
    # page advertises a threshold, so assert the threshold itself -- on the same
    # seed and the same honest strategy, only sigma changes sides.
    def _honest_errors(sigma):
        """Round-2 |p - v| for an honest firm (a=1, m=0) under each rule."""
        out = {}
        for mode in ("memory", "static"):
            sp = Space(seed=2, mode=mode, sigma=sigma)
            sp.submit(1, 0)
            r = sp.submit(1, 0)
            out[mode] = abs(r.p - r.v)
        return out["memory"], out["static"]

    rev_lo, com_lo = _honest_errors(0.5)          # sigma < m_bar
    assert rev_lo < com_lo, "below sigma = m_bar the revisable rule must help the honest firm"

    rev_hi, com_hi = _honest_errors(2.0)          # sigma > m_bar
    assert rev_hi > com_hi, "above sigma = m_bar it must hurt -- this is the paper's threshold"
    return True
