"""
run_experiments.py -- reproduces every number the PS1 paper reports.

    python src/run_experiments.py

Writes CSVs and figures into results/ and prints the headline table.
Deterministic: fixed seeds, no wall-clock, no network.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategic_disclosure import (  # noqa: E402
    KAPPAS, LAMBDA, M_BAR, THETAS, WD_GRID, Space, best_response,
    false_punishment_rate, verify_space_replication, verify_vectorisation,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TYPES = list(itertools.product(THETAS, KAPPAS))
RULES = ["committed", "revisable"]


def _aggregate(rows: list[dict]) -> dict:
    d = pd.DataFrame(rows)
    return {
        "honest_share": d["honest"].mean(),
        "manipulation": d["manipulation"].mean(),
        "mispricing": d["mispricing"].mean(),
        "welfare": d["welfare"].mean(),
        "deviation_gain": d["deviation_gain"].mean(),
        "max_deviation_gain": d["deviation_gain"].max(),
    }


def sweep_noise() -> pd.DataFrame:
    """Reviewer 1: 'test the model under different levels of outcome noise'."""
    out = []
    for sigma, rule in itertools.product(SIGMAS, RULES):
        rows = [best_response(t, k, rule, sigma, LAMBDA, seed=11) for t, k in TYPES]
        out.append({"sigma": sigma, "rule": rule, **_aggregate(rows)})
    return pd.DataFrame(out)


def sweep_lambda(sigma: float) -> pd.DataFrame:
    """Reviewer 2: 'test ... memory strength before drawing conclusion'."""
    out = []
    for lam in LAMBDAS:
        rows = [best_response(t, k, "revisable", sigma, lam, seed=11) for t, k in TYPES]
        out.append({"lambda": lam, "sigma": sigma, "rule": "revisable", **_aggregate(rows)})
    rows = [best_response(t, k, "committed", sigma, LAMBDA, seed=11) for t, k in TYPES]
    out.append({"lambda": float("nan"), "sigma": sigma, "rule": "committed", **_aggregate(rows)})
    return pd.DataFrame(out)


def sweep_false_punishment() -> pd.DataFrame:
    """Reviewer 1: 'Can memory sometimes incorrectly punish an honest firm?'"""
    return pd.DataFrame([
        false_punishment_rate(s, lam)
        for lam in (0.5, 1.0, 1.5) for s in SIGMAS
    ])


def seed_audit(n_seeds: int = 500) -> pd.DataFrame:
    """How rare was the v1 accident that eps drew 0.00 in every priced round?"""
    rows = []
    for seed in range(n_seeds):
        sp = Space(seed=seed, mode="memory")
        eps = [sp.submit(1, 0).eps for _ in range(2)]
        rows.append({"seed": seed, "theta": sp.theta, "k": sp.k,
                     "eps1": eps[0], "eps2": eps[1],
                     "both_zero": float(eps[0] == 0.0 and eps[1] == 0.0)})
    return pd.DataFrame(rows)


# The browser grid is coarse on purpose (three window-dressing levels, three
# cost types) so the Space stays playable.  v1 promised to report the difference
# between that grid and a research grid rather than hide it; this is that report.
RESEARCH_WD = tuple(np.round(np.arange(0.0, 2.01, 0.25), 2))
RESEARCH_KAPPAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def sweep_grid_sensitivity(sigma: float = 0.5) -> pd.DataFrame:
    """Coarse browser grid vs. a finer research grid, same rules, same noise."""
    out = []
    specs = [("browser", list(TYPES), tuple(WD_GRID)),
             ("research", list(itertools.product(THETAS, RESEARCH_KAPPAS)), RESEARCH_WD)]
    for grid_name, types, wd in specs:
        for rule in RULES:
            rows = [best_response(t, k, rule, sigma, LAMBDA, seed=11, wd_grid=wd)
                    for t, k in types]
            out.append({"grid": grid_name, "n_types": len(types), "n_wd": len(wd),
                        "rule": rule, "sigma": sigma, **_aggregate(rows)})
    return pd.DataFrame(out)


def per_type_detail(sigma: float = 0.5) -> pd.DataFrame:
    """Which firm types inflate, and under which rule."""
    rows = []
    for (t, k), rule in itertools.product(TYPES, RULES):
        rows.append(best_response(t, k, rule, sigma, LAMBDA, seed=11))
    return pd.DataFrame(rows)[
        ["theta", "k", "rule", "a1", "m1", "a2", "m2",
         "payoff", "mispricing", "welfare", "deviation_gain"]
    ]


def main() -> None:
    assert verify_space_replication(), "Python twin diverged from the browser Space"
    assert verify_vectorisation(), "vectorised expectation diverged from the scalar model"
    print("verification: Python twin reproduces all four logged browser cases  [OK]")

    noise = sweep_noise();                 noise.to_csv(f"{RESULTS}/sweep_noise.csv", index=False)
    lam05 = sweep_lambda(0.5);             lam05.to_csv(f"{RESULTS}/sweep_lambda_sigma0.5.csv", index=False)
    lam15 = sweep_lambda(1.5);             lam15.to_csv(f"{RESULTS}/sweep_lambda_sigma1.5.csv", index=False)
    fp = sweep_false_punishment();         fp.to_csv(f"{RESULTS}/false_punishment.csv", index=False)
    seeds = seed_audit();                  seeds.to_csv(f"{RESULTS}/seed_audit.csv", index=False)
    grids = sweep_grid_sensitivity();      grids.to_csv(f"{RESULTS}/grid_sensitivity.csv", index=False)
    detail = per_type_detail();            detail.to_csv(f"{RESULTS}/per_type_detail.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30,
                  "display.float_format", lambda x: f"{x:7.3f}")

    print("\n=== A. Noise sweep, lambda = 1, averaged over all 9 firm types ===")
    print(noise.pivot(index="sigma", columns="rule",
                      values=["honest_share", "mispricing", "welfare", "deviation_gain"]))

    print("\n=== B. Can memory mis-punish an honest firm? (m = 0, analytic) ===")
    print(fp[fp["lambda"] == 1.0].to_string(index=False))

    print("\n=== C. Memory strength lambda, sigma = 0.5 ===")
    print(lam05.to_string(index=False))

    print("\n=== D. Memory strength lambda, sigma = 1.5 ===")
    print(lam15.to_string(index=False))

    print("\n=== E. Seed audit: share of seeds with eps = 0 in BOTH priced rounds ===")
    print(f"{seeds['both_zero'].mean():.3f} of {len(seeds)} seeds "
          f"(seed 2026, used in v1, is one of them)")

    print("\n=== F. Grid sensitivity: coarse browser grid vs. research grid ===")
    print(grids.to_string(index=False))

    print("\n=== G. Per-type detail, sigma = 0.5, lambda = 1 ===")
    print(detail.to_string(index=False))

    _figures(noise, fp, lam05)
    print(f"\nwrote CSVs and figures to {RESULTS}/")


def _figures(noise: pd.DataFrame, fp: pd.DataFrame, lam: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))

    for rule, style in (("committed", "-o"), ("revisable", "-s")):
        d = noise[noise["rule"] == rule]
        ax[0].plot(d["sigma"], d["mispricing"], style, ms=3.5, label=rule)
        ax[2].plot(d["sigma"], d["deviation_gain"], style, ms=3.5, label=rule)
    ax[0].set(xlabel=r"outcome noise $\sigma$", ylabel=r"mean $|p-v|$",
              title="Disclosure accuracy")
    ax[0].axvline(2 * M_BAR, color="0.6", lw=0.8, ls=":")
    ax[0].legend(frameon=False)

    d = fp[fp["lambda"] == 1.0]
    ax[1].plot(d["sigma"], d["worse_than_committed_share"], "-^", ms=3.5, color="#b45309")
    ax[1].axvline(2 * M_BAR, color="0.6", lw=0.8, ls=":")
    ax[1].set(xlabel=r"outcome noise $\sigma$",
              ylabel="share of honest firms priced\nworse than under commitment",
              title="Mis-punishment of the honest")

    ax[2].set(xlabel=r"outcome noise $\sigma$", ylabel="mean deviation gain",
              title="Gain from inflating")
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig_noise.png", bbox_inches="tight")
    fig.savefig(f"{RESULTS}/fig_noise.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    d = lam[lam["rule"] == "revisable"]
    ax.plot(d["lambda"], d["mispricing"], "-s", ms=3.5, label="revisable")
    base = lam[lam["rule"] == "committed"]["mispricing"].iloc[0]
    ax.axhline(base, color="0.4", ls="--", lw=1, label="committed")
    ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set(xlabel=r"memory strength $\lambda$", ylabel=r"mean $|p-v|$",
           title=r"$\sigma = 0.5$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig_lambda.png", bbox_inches="tight")
    fig.savefig(f"{RESULTS}/fig_lambda.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
