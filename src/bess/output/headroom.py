"""Hindsight headroom of the learned value: how much could any end-of-day (and dispatch-mode) policy gain?

Development years only (2024 and 2025); the 2026 test is not used. Usage:
  python -m bess benchmark headroom                 # end-level headroom from the reward caches (both profiles)
  python -m bess benchmark headroom --hurdle-menu   # also the ISEA battery with a dispatch menu (new MILPs)

For each year the chain starts at 50 % and restarts there at missing days. With realised rewards from the cache:
  fixed_50      end every day at 50 % (the fixed-end baseline)
  best_const    the best single end level (and mode) used every day
  myopic        best end level for today only, in hindsight
  two_day       best for today plus the best possible tomorrow, in hindsight (executes today, repeats)
  dp            full-chain dynamic programme in hindsight (upper bound for every end-level policy)
`dp - best_const` is all an end-level policy could gain over the best simple rule; `dp - two_day` is what looking
beyond tomorrow (multi-step bootstrapping, gamma > 0) could add over a one-successor value.
Dispatch menu (--hurdle-menu): the daily MILP is planned with an extra discharge cost h in {0, 10, 20} EUR/MWh and
settled without it; levels {0, 20, 50, 80, 100 %}. Writes <results_dir>/headroom.csv for each profile.
"""
from __future__ import annotations

import argparse
import dataclasses
from multiprocessing import get_context

import numpy as np
import polars as pl

from bess.core import calendar as cal
from bess.core.config import forecaster_path, v2_battery, v2_config, v2_dir
from bess.core.markets.day_ahead import DayAheadMarket
from bess.forecasting.forward_fold import ForwardFoldXGB
from bess.input.markets import resolve_market
from bess.input.prices import load_market_data
from bess.milp.model import schedule_objective, solve_schedule
from bess.rl.value.cache import (
    gate_closure_inputs,
    mid_index,
    reward_tensor,
    soc_levels,
)

YEARS = (2024, 2025)
MENU_LEVELS = (0.0, 0.2, 0.5, 0.8, 1.0)
HURDLES = (0.0, 10.0, 20.0)


def segments(days: list) -> list[list[int]]:
    segs, cur = [], []
    for i, d in enumerate(days):
        if cur and (d - days[i - 1]).days != 1:
            segs.append(cur)
            cur = []
        cur.append(i)
    return segs + [cur]


def chain_values(days: list, R: list[np.ndarray], s0: int) -> dict:
    """R[k]: (n_start, n_end, n_mode) realised reward of day k."""
    n, nm = R[0].shape[1], R[0].shape[2]
    segs = segments(days)

    def run(choose):
        tot = 0.0
        for sg in segs:
            s = s0
            for p, k in enumerate(sg):
                e, m = choose(sg, p, k, s)
                tot += R[k][s, e, m]
                s = e
        return tot

    consts = {(e, m): run(lambda sg, p, k, s, em=(e, m): em) for e in range(n) for m in range(nm)}
    best = max(consts, key=consts.get)

    def two_day(sg, p, k, s):
        Q = R[k][s].copy()
        if p + 1 < len(sg):
            Q += R[sg[p + 1]].max(axis=(1, 2))[:, None]
        return np.unravel_index(np.argmax(Q), (n, nm))

    dp = 0.0
    for sg in segs:
        V = np.zeros(n)
        for k in reversed(sg):
            V = (R[k] + V[None, :, None]).max(axis=(1, 2))
        dp += V[s0]
    return {"fixed_50": consts[(s0, 0)], "best_const": consts[best], "best_const_end": best[0], "best_const_mode": best[1],
            "myopic": run(lambda sg, p, k, s: np.unravel_index(np.argmax(R[k][s]), (n, nm))),
            "two_day": run(two_day), "dp": dp}


def rows_for(name: str, days: list, R: list, s0: int, levels, modes) -> list[dict]:
    out = []
    for y in YEARS:
        idx = [i for i, d in enumerate(days) if d.year == y]
        if not idx:
            continue
        v = chain_values([days[i] for i in idx], [R[i] for i in idx], s0)
        nd = len(idx)
        out.append({"analysis": name, "year": y, "n_days": nd, **{k: v[k] for k in ("fixed_50", "best_const", "myopic", "two_day", "dp")},
                    "best_const_end_soc": float(levels[v["best_const_end"]]), "best_const_mode": modes[v["best_const_mode"]],
                    "dp_minus_best_const_eur_day": (v["dp"] - v["best_const"]) / nd,
                    "dp_minus_two_day_eur_day": (v["dp"] - v["two_day"]) / nd,
                    "myopic_minus_best_const_eur_day": (v["myopic"] - v["best_const"]) / nd})
    return out


# ------------------------------------------------------------------ dispatch menu (new MILPs)
_W: dict = {}


def _init(b):
    _W["b"], _W["m"] = b, DayAheadMarket()


def _menu_day(args):
    d, res, f, actual = args
    b, m = _W["b"], _W["m"]
    dt_h = cal.RESOLUTION_HOURS[res]
    R = np.empty((len(MENU_LEVELS), len(MENU_LEVELS), len(HURDLES)))
    for k, h in enumerate(HURDLES):
        bp = dataclasses.replace(b, c_deg_eur_per_mwh=b.c_deg_eur_per_mwh + h)   # planning only
        for i, s in enumerate(MENU_LEVELS):
            for j, e in enumerate(MENU_LEVELS):
                r = solve_schedule(f, dt_h, bp, [m], e_init=s * b.e_max_mwh, e_final=e * b.e_max_mwh)
                R[i, j, k] = schedule_objective(r.schedule, actual, b, [m])          # settled without the hurdle
    return d, R


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hurdle-menu", action="store_true")
    args = ap.parse_args(argv)
    for config in ("v2_isea.yaml", "v2.yaml"):
        v2, base = v2_config(config)
        battery = v2_battery(v2)
        levels = soc_levels(battery)
        cache = pl.read_parquet(v2_dir(v2, "cache") / "reward_cache.parquet")
        days = sorted(d for d in cache["day"].unique().to_list() if d.year in YEARS)
        R = reward_tensor(cache, days)[..., None]
        rows = rows_for("end_level", days, list(R), mid_index(levels), levels, (0.0,))
        if args.hurdle_menu and config == "v2_isea.yaml":
            data = load_market_data(base)
            fc = ForwardFoldXGB.load(forecaster_path(v2))
            spec = resolve_market(base)
            inp = gate_closure_inputs(data, fc, days, spec.qh_start_date, spec.gate_closure_local,
                                      spec.publication_local)
            jobs = [(d, r, f, data.series(r).day_prices(d)) for d, (r, f, _) in sorted(inp.items())]
            with get_context("spawn").Pool(v2["cache"]["n_procs"], initializer=_init, initargs=(battery,)) as p:
                res = dict(p.map(_menu_day, jobs, chunksize=4))
            md = sorted(res)
            rows += rows_for("end_level_x_discharge_hurdle", md, [res[d] for d in md], MENU_LEVELS.index(0.5),
                             MENU_LEVELS, HURDLES)
        out = pl.DataFrame(rows)
        out.write_csv(v2_dir(v2, "results_dir") / "headroom.csv", float_precision=3)
        with pl.Config(tbl_cols=-1, tbl_width_chars=220, float_precision=2, tbl_hide_dataframe_shape=True):
            print(config)
            print(out.select("analysis", "year", "n_days", "fixed_50", "best_const", "two_day", "dp",
                             "dp_minus_best_const_eur_day", "dp_minus_two_day_eur_day"))


if __name__ == "__main__":
    main()
