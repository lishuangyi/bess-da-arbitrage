"""Benchmark protocol: decide which method is best, with uncertainty.

Rules (fixed before looking at test results):
1. Decision period = the sealed test period (15-min). Validation is reported for
   reference only (it was used for model selection / early stopping).
2. All methods are compared on the identical set of delivery days.
3. Gate: a method with any constraint violation is disqualified.
4. Primary metric: total net revenue (revenue - degradation), equivalently the
   capture rate w.r.t. perfect foresight. Oracle methods (perfect foresight,
   incl. its hourly-block variant) are reference bounds and are not ranked.
5. Uncertainty: moving-block bootstrap over days (block = 7 days, to respect
   weekly autocorrelation) of the *paired* daily differences to the leader.
   A method is "tied with the leader" if the 95 % CI of its mean daily
   difference to the leader contains 0. Tier 1 = leader + all tied methods.
6. Secondary (reported, not used for the rank): worst-5 % day mean, loss-day
   share, cycles per day, revenue per cycle, months won, compute time P95.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import polars as pl

from bess.backtest.metrics import summarize

ORACLE_PREFIX = "perfect_foresight"


def block_indices(n: int, block: int = 7, n_boot: int = 5000, seed: int = 0) -> np.ndarray:
    """(n_boot, n) day indices of moving-block bootstrap resamples (contiguous blocks of `block` days)."""
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]


def block_bootstrap_means(x: np.ndarray, block: int = 7, n_boot: int = 5000, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap distribution of the mean of a daily series."""
    x = np.asarray(x, float)
    return x[block_indices(len(x), block, n_boot, seed)].mean(axis=1)


def block_bootstrap_ratio(num: np.ndarray, den: np.ndarray, block: int = 7, n_boot: int = 5000,
                          seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of sum(num) / sum(den), resampling the paired days together."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = block_indices(len(num), block, n_boot, seed)
    return num[idx].sum(axis=1) / den[idx].sum(axis=1)


def ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(dist, a)), float(np.quantile(dist, 1 - a))


def _on_days(df: pl.DataFrame, days: list) -> pl.DataFrame:
    return pl.DataFrame({"day": days}).join(df, on="day", how="left")


def run_benchmark(daily: dict[str, pl.DataFrame], reference: pl.DataFrame, block: int = 7,
                  n_boot: int = 5000, seed: int = 0) -> pl.DataFrame:
    """`daily`: method -> per-day results; `reference`: perfect-foresight per-day results."""
    days = set(reference["day"].to_list())
    for df in daily.values():
        days &= set(df["day"].to_list())
    days = sorted(days)
    ref = _on_days(reference, days)
    ref_net = ref["net_eur"].to_numpy()
    nets = {m: _on_days(df, days)["net_eur"].to_numpy() for m, df in daily.items()}
    months = [(d.year, d.month) for d in days]
    rows = []
    for m, df in daily.items():
        d = _on_days(df, days)
        s = summarize(d, ref)
        # numerator and denominator resampled on the same days
        cap = block_bootstrap_ratio(nets[m], ref_net, block, n_boot, seed) * 100.0
        lo, hi = ci(cap)
        rows.append({"method": m, "oracle": m.startswith(ORACLE_PREFIX), "n_days": len(days),
                     "disqualified": s["violations"] > 0, **{k: v for k, v in s.items() if k != "n_days"},
                     "capture_ci95_lo": lo, "capture_ci95_hi": hi,
                     "eur_per_cycle": s["net_revenue_eur"] / max(1e-9, float(d["efc"].sum()))})
    ranked = sorted((r for r in rows if not r["oracle"] and not r["disqualified"]),
                    key=lambda r: r["net_revenue_eur"], reverse=True)
    for r in rows:
        r.update(rank=None, diff_to_leader_eur_per_day=None, diff_ci95_lo=None, diff_ci95_hi=None,
                 months_won=0, tier="reference" if r["oracle"] else ("disqualified" if r["disqualified"] else ""))
    if ranked:
        leader = ranked[0]["method"]
        # months won: best monthly net revenue among ranked methods
        names = [r["method"] for r in ranked]
        monthly = {m: Counter() for m in names}
        for m in names:
            for key, v in zip(months, nets[m]):
                monthly[m][key] += v
        wins = Counter(max(names, key=lambda m: monthly[m][k]) for k in sorted(set(months)))
        for i, r in enumerate(ranked, start=1):
            m = r["method"]
            diff = nets[m] - nets[leader]
            lo, hi = (0.0, 0.0) if m == leader else ci(block_bootstrap_means(diff, block, n_boot, seed))
            r.update(rank=i, diff_to_leader_eur_per_day=float(diff.mean()), diff_ci95_lo=lo, diff_ci95_hi=hi,
                     months_won=int(wins.get(m, 0)),
                     tier="1 (leader)" if m == leader else ("1 (tied)" if lo <= 0 <= hi else "2+"))
    rows.sort(key=lambda r: (not r["oracle"], r["rank"] if r["rank"] is not None else 10**9))
    return pl.DataFrame(rows)
