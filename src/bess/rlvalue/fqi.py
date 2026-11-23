"""Fitted Q iteration for the end-of-day SoC level (v2).

MDP, one step per delivery day: state s_D = (phi_D, start level e_s), action a_D = end level e_e
(9 levels each, the next day starts at today's end level), reward R[D, e_s, e_e] from the reward
cache (daily MILP on the gate-closure forecast with fixed start and end, settled at realised
prices). Every action is in the data set for every day.

    Q_0  = fit(R)
    Q_k  = fit(R[D, s, e] + gamma * max_a' Q_{k-1}(phi_{D+1}, e, a'))     (terminal if D+1 is missing)

Rewards are divided by `scale` (mean daily perfect-foresight revenue of the training days).
With gamma = 0 the fit is the one-step model (no bootstrapping), used for the B3 ablation.

Continuation value of ending day D at level e:  V(phi_{D+1}, e) = max_a' Q(phi_{D+1}, e, a').
phi_{D+1} is only known at gate closure of D+1, so a projection W(phi_D, e) is fitted to
V(phi_{D+1}, e) - V(phi_{D+1}, 50 %) on the training days; the MILP of day D then adds W(phi_D, e)
(in EUR) for the end level it chooses (bess.rlvalue.policy.LearnedValueMILP). The weight of W in
the MILP is 1 for every gamma, so gamma = 0 gives a one-day look-ahead value (B3) and gamma > 0 the
multi-step value; with a weight of gamma, gamma = 0 would drop the value and equal B1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from bess.rlvalue.cache import mid_index, reward_tensor
from bess.rlvalue.features import FEATURE_NAMES


@dataclass
class FQIData:
    days: list
    X: np.ndarray          # (n, n_features) phi_D
    R: np.ndarray          # (n_days, n, n) reward in EUR
    next_idx: np.ndarray   # (n_days,) index of D+1 in `days`, -1 if D+1 is not in the set (chain end)
    levels: np.ndarray     # (n,) SoC fractions of the levels


def load_fqi_data(cache: pl.DataFrame, features: pl.DataFrame, start, end, levels: np.ndarray) -> FQIData:
    f = features.filter((pl.col("day") >= start) & (pl.col("day") <= end)).sort("day")
    have = set(cache.filter((pl.col("day") >= start) & (pl.col("day") <= end))["day"].unique().to_list())
    f = f.filter(pl.col("day").is_in(list(have)))
    days = f["day"].to_list()
    pos = {d: i for i, d in enumerate(days)}
    import datetime as dt
    nxt = np.array([pos.get(d + dt.timedelta(days=1), -1) for d in days])
    R = reward_tensor(cache, days)
    if R.shape[1] != len(levels):
        raise ValueError(f"cache has {R.shape[1]} levels, battery profile has {len(levels)}")
    return FQIData(days, f.select(FEATURE_NAMES).to_numpy(), R, nxt, np.asarray(levels, float))


def q_inputs(X: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Rows (day, s, e) in C order: [phi_D, level_s, level_e]."""
    n, N = len(X), len(levels)
    s = np.repeat(np.tile(levels, n), N)
    e = np.tile(levels, n * N)
    return np.column_stack([np.repeat(X, N * N, axis=0), s, e])


def w_inputs(X: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Rows (day, e): [phi_D, level_e]."""
    return np.column_stack([np.repeat(X, len(levels), axis=0), np.tile(levels, len(X))])


def make_regressor(kind: str, params: dict, seed: int):
    if kind == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=6, **params)
    if kind == "hgb":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(kind)


@dataclass
class FQIResult:
    gamma: float
    scale: float
    models: dict                     # iteration k -> fitted Q model (snapshots)
    deltas: list = field(default_factory=list)   # mean |Q_k - Q_{k-1}| on the training rows (scaled units)


def fit_fqi(d: FQIData, gamma: float, iterations: int, kind: str, params: dict, seed: int, scale: float,
            snapshots: tuple = (), verbose: bool = False) -> FQIResult:
    Xq = q_inputs(d.X, d.levels)
    N = len(d.levels)
    R = d.R / scale
    has_next = d.next_idx >= 0
    K = 1 if gamma == 0 else iterations
    keep = set(snapshots) | {K}
    res = FQIResult(gamma, scale, {})
    q_prev = None
    for k in range(1, K + 1):
        target = R.copy()
        if q_prev is not None:
            v_next = q_prev.max(axis=2)                      # (n_days, n): value of starting day j at level e
            target[has_next] += gamma * v_next[d.next_idx[has_next]][:, None, :]
        m = make_regressor(kind, params, seed).fit(Xq, target.reshape(-1))
        q = m.predict(Xq).reshape(-1, N, N)
        if q_prev is not None:
            res.deltas.append(float(np.mean(np.abs(q - q_prev))))
            if verbose:
                print(f"    FQI gamma={gamma} iteration {k}: mean |dQ| = {res.deltas[-1]:.4f}", flush=True)
        q_prev = q
        if k in keep:
            res.models[k] = m
    return res


def continuation_values(q_model, X: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """V(phi, e) = max_a Q(phi, e, a) for every row of X: (n_rows, n_levels)."""
    N = len(levels)
    return q_model.predict(q_inputs(X, levels)).reshape(-1, N, N).max(axis=2)


def fit_projection(d: FQIData, q_model, kind: str, params: dict, seed: int):
    """W(phi_D, e) ~ V(phi_{D+1}, e) - V(phi_{D+1}, 50 %) on days whose successor is known (scaled units)."""
    has_next = d.next_idx >= 0
    V = continuation_values(q_model, d.X[d.next_idx[has_next]], d.levels)
    T = V - V[:, [mid_index(d.levels)]]
    return make_regressor(kind, params, seed).fit(w_inputs(d.X[has_next], d.levels), T.reshape(-1))
