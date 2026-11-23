"""Reward cache for the end-of-day SoC value (v2): n x n start/end SoC levels per delivery day
(n = 9 for the project battery, 11 for the ISEA battery; see soc_levels).

For every delivery day D and every pair (start level e_s, end level e_e), the daily MILP is
solved on the gate-closure forecast of D, starting at e_s and ending exactly at e_e. The
resulting schedule is settled at the realised prices of D:
  R[D, s, e] = realised net revenue (revenue - degradation cost)      -> the reward
  F[D, s, e] = objective value under the forecast                     -> what the MILP expected
All n x n actions are computed for every day, so the offline data set covers the whole action
space (no out-of-distribution actions in fitted Q iteration).
"""
from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import time

import numpy as np
import polars as pl

from bess.config import Battery
from bess.data import calendar as cal
from bess.data.smard import MarketData
from bess.markets.day_ahead import DayAheadMarket
from bess.optim.milp import count_violations, schedule_objective, solve_schedule
from bess.rlvalue.features import FEATURE_NAMES, state_features

LEVEL_STEP = 0.1   # SoC levels every 10 percentage points between soc_min and soc_max


def soc_levels(battery: Battery) -> np.ndarray:
    """SoC fractions of the levels: 10 %..90 % (9 levels) for the project battery, 0..100 % (11) for ISEA's."""
    n = int(round((battery.soc_max - battery.soc_min) / LEVEL_STEP)) + 1
    lv = np.round(battery.soc_min + LEVEL_STEP * np.arange(n), 10)
    if abs(lv[-1] - battery.soc_max) > 1e-9:
        raise ValueError("the SoC range is not a multiple of LEVEL_STEP")
    return lv


def mid_index(levels: np.ndarray) -> int:
    """Index of the 50 % level (the restart level of every chain)."""
    i = int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
    if abs(levels[i] - 0.5) > 1e-9:
        raise ValueError("no 50 % level")
    return i


def level_energy(battery: Battery) -> np.ndarray:
    """Energy (MWh) of each SoC level."""
    return soc_levels(battery) * battery.e_max_mwh


def resolution_of(day, qh_start: dt.date) -> str:
    return "quarterhour" if cal.to_date(day) >= qh_start else "hour"


def gate_closure_inputs(data: MarketData, forecaster, days: list, qh_start: dt.date, gc: str, pub: str) -> dict:
    """{D: (resolution, f_D, phi_D)} with the forecasts of D and D+1 made at gate closure of D.
    Days whose forecasts cannot be made (no model, missing lags) are left out."""
    out = {}
    for d in days:
        info = data.known_at(cal.gate_closure_utc(d, gc), pub)
        res0 = resolution_of(d, qh_start)
        nxt = d + dt.timedelta(days=1)
        res1 = resolution_of(nxt, qh_start)
        try:
            f0 = forecaster.predict(d, info, res0)       # horizon 1 (D-1 is the last known day)
            f1 = forecaster.predict(nxt, info, res1)     # horizon 2
        except (ValueError, RuntimeError):
            continue
        out[d] = (res0, f0, state_features(d, f0, cal.RESOLUTION_HOURS[res0], f1, cal.RESOLUTION_HOURS[res1]))
    return out


# ------------------------------------------------------------------ workers
_W: dict = {}


def _init_worker(battery: Battery):
    _W["battery"] = battery
    _W["market"] = DayAheadMarket()
    _W["levels"] = level_energy(battery)


def solve_day(args) -> list[dict]:
    """All (start, end) MILPs of one day: forecast f, settled at realised prices `actual`."""
    day, res, f, actual = args
    b, mk, lv = _W["battery"], _W["market"], _W["levels"]
    dt_h = cal.RESOLUTION_HOURS[res]
    rows = []
    for s, e0 in enumerate(lv):
        for e, e1 in enumerate(lv):
            r = solve_schedule(f, dt_h, b, [mk], e_init=float(e0), e_final=float(e1))
            sch = r.schedule
            rows.append({"day": day, "s": s, "e": e, "net_eur": schedule_objective(sch, actual, b, [mk]),
                         "forecast_obj_eur": r.objective, "discharge_mwh": float(sch.p_dis.sum() * dt_h),
                         "violations": count_violations(sch, b, float(e0), float(e1)),
                         "solve_ms": 1e3 * r.solve_time_s})
    return rows


def build_cache(data: MarketData, inputs: dict, battery: Battery, n_procs: int = 6,
                progress_every: int = 50) -> pl.DataFrame:
    """Reward table for every day in `inputs` (see gate_closure_inputs)."""
    jobs = []
    for d, (res, f, _) in sorted(inputs.items()):
        actual = data.series(res).day_prices(d)
        if np.isfinite(actual).all():
            jobs.append((d, res, f, actual))
    rows, t0 = [], time.perf_counter()
    ctx = mp.get_context("spawn")   # polars is not fork-safe
    with ctx.Pool(n_procs, initializer=_init_worker, initargs=(battery,)) as pool:
        for i, day_rows in enumerate(pool.imap(solve_day, jobs, chunksize=4), start=1):
            rows += day_rows
            if progress_every and (i == 10 or i % progress_every == 0 or i == len(jobs)):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)} days, {el / 60:.1f} min, about {el / i * (len(jobs) - i) / 60:.1f} min left",
                      flush=True)
    return pl.DataFrame(rows)


def features_frame(inputs: dict) -> pl.DataFrame:
    days = sorted(inputs)
    X = np.vstack([inputs[d][2] for d in days])
    return pl.DataFrame({"day": days, "resolution": [inputs[d][0] for d in days],
                         **{n: X[:, j] for j, n in enumerate(FEATURE_NAMES)}})


def reward_tensor(cache: pl.DataFrame, days: list, column: str = "net_eur") -> np.ndarray:
    """(len(days), n, n) array of `column` indexed [day, start level, end level]."""
    c = cache.filter(pl.col("day").is_in(days)).sort(["day", "s", "e"])
    n = int(cache["e"].max()) + 1
    if c.height != n * n * len(days):
        raise ValueError(f"cache has {c.height} rows for {len(days)} days (expected {n * n} per day)")
    pos = {d: i for i, d in enumerate(days)}
    out = np.empty((len(days), n, n))
    out[[pos[d] for d in c["day"].to_list()], c["s"].to_numpy(), c["e"].to_numpy()] = c[column].to_numpy()
    return out
