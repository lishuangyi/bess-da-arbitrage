"""Fit the end-of-day SoC value by fitted Q iteration and select its hyper-parameters on validation (v2).

Usage:
  python scripts/train_fqi.py              # grid on validation, seeds, final refit
  python scripts/train_fqi.py --quick      # small grid (smoke test)

1. scale = mean daily perfect-foresight net revenue of the training days (fixed 50 % per day).
2. Grid on the training period (2021-2024): gamma in {0, 0.9, 0.95, 0.98}, tree settings, and
   iterations K (snapshots); for every candidate the projection W is fitted and M1 (MILP + learned
   terminal value) is run as a chained backtest on the validation period (2025, incl. the 15-min
   Q4). Selection by validation net revenue incl. terminal valuation. gamma = 0 is the one-step
   value (B3); it uses the tree setting chosen for M1, so that M1 and B3 differ only in gamma.
3. The chosen M1 and B3 settings are fitted with seeds 0, 1, 2 (ExtraTrees random_state) and run on
   validation; then refitted on training + validation (2021-2025, no 2026 data) for the test and
   acceptance periods.
Writes <results_dir>/fqi_grid.csv, fqi_seeds_validation.csv, fqi_convergence.csv, fqi_choice.json,
<results_dir>/daily/V2__<method>__validation__seed<k>.csv, and <models_dir>/fqi_<role>_seed<k>.pkl.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import polars as pl

from bess.backtest.chain import mean_period_price, run_chain
from bess.backtest.engine import settle_day
from bess.data.smard import load_market_data
from bess.markets.day_ahead import DayAheadMarket
from bess.optim.milp import PerfectForesight
from bess.rlvalue.fqi import fit_fqi, fit_projection, load_fqi_data
from bess.rlvalue.forecasts import ForwardFoldXGB
from bess.rlvalue.cache import soc_levels
from bess.rlvalue.periods import add_v2_config_arg, forecaster_path, period_days_res, qh_start, v2_battery, v2_config, v2_dir
from bess.rlvalue.policy import LearnedEndLevelMILP, LearnedValueMILP

GAMMAS = [0.0, 0.9, 0.95, 0.98]
TREES = {"et_leaf20": {"n_estimators": 50, "min_samples_leaf": 20, "max_features": 0.5},
         "et_leaf80": {"n_estimators": 50, "min_samples_leaf": 80, "max_features": 0.5}}
K_SNAPSHOTS = (10, 20, 30)
SEEDS = (0, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="gamma in {0, 0.95}, one tree setting, K in {5, 10}")
    ap.add_argument("--only-b3", action="store_true",
                    help="keep the grid and the M1 choice; redo B3 (gamma = 0 with M1's tree setting) only")
    add_v2_config_arg(ap)
    args = ap.parse_args()
    v2, base = v2_config(args.config)
    pub = base["day_ahead"]["publication_local"]
    market = DayAheadMarket(base["day_ahead"]["gate_closure_local"])
    data = load_market_data(base)
    battery = v2_battery(v2)
    levels = soc_levels(battery)
    q0 = qh_start(base)
    models, out, cache_dir = v2_dir(v2, "models_dir"), v2_dir(v2, "results_dir"), v2_dir(v2, "cache")
    (out / "daily").mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    fc = ForwardFoldXGB.load(forecaster_path(v2))
    cache = pl.read_parquet(cache_dir / "reward_cache.parquet")
    feats = pl.read_parquet(cache_dir / "features.parquet")
    P = v2["periods"]
    tr = load_fqi_data(cache, feats, *map(_d, (P["train"]["start"], P["train"]["end"])), levels)
    trva = load_fqi_data(cache, feats, *map(_d, (P["train"]["start"], P["validation"]["end"])), levels)
    val_days, _ = period_days_res(v2, base, data, "validation")
    p_bar = mean_period_price(data, val_days)
    print(f"FQI training days: {len(tr.days)} ({int((tr.next_idx >= 0).sum())} with a successor); "
          f"validation chain: {len(val_days)} days", flush=True)

    # 1. reward scale from the training days
    scale_path = out / "fqi_scale.json"
    if scale_path.exists():
        scale = json.loads(scale_path.read_text())["scale_eur_per_day"]
    else:
        t0 = time.perf_counter()
        train_res, _ = period_days_res(v2, base, data, "train")
        pf = [settle_day(PerfectForesight(), data, d, r, battery.e_init, battery, market, pub)["net_eur"]
              for d, r in train_res]
        scale = float(np.mean(pf))
        scale_path.write_text(json.dumps({"scale_eur_per_day": scale, "n_days": len(pf),
                                          "definition": "mean daily perfect-foresight net revenue, training days"}))
        print(f"scale = {scale:.1f} EUR/day ({time.perf_counter() - t0:.0f} s)", flush=True)

    gammas = [0.0, 0.95] if args.quick else GAMMAS
    trees = dict(list(TREES.items())[:1]) if args.quick else TREES
    snaps = (5, 10) if args.quick else K_SNAPSHOTS

    def val_chain(policy, key=None):
        df = run_chain(policy, data, val_days, battery, market, pub, mean_price=p_bar)
        if key:
            df.write_csv(out / "daily" / key)
        return df

    # 2. grid with seed 0
    grid, conv = [], []
    if args.only_b3:
        gammas = []
    for g in gammas:
        for tname, tparams in trees.items():
            t0 = time.perf_counter()
            res = fit_fqi(tr, g, max(snaps), "extra_trees", tparams, seed=0, scale=scale, snapshots=snaps)
            conv += [{"gamma": g, "trees": tname, "iteration": i + 2, "mean_abs_dq": v} for i, v in enumerate(res.deltas)]
            for K, qm in sorted(res.models.items()):
                w = fit_projection(tr, qm, "extra_trees", tparams, seed=0)
                df = val_chain(LearnedValueMILP(fc, w, scale, q0))
                grid.append({"gamma": g, "trees": tname, "K": K, "val_net_adj_eur": float(df["net_adj_eur"].sum()),
                             "val_net_eur": float(df["net_eur"].sum()), "efc_per_day": float(df["efc"].mean()),
                             "violations": int(df["violations"].sum())})
                print(f"  gamma={g:<5} {tname} K={K:<3} validation net+terminal = {grid[-1]['val_net_adj_eur']:10.0f} EUR "
                      f"(fit {time.perf_counter() - t0:.0f} s)", flush=True)
    if args.only_b3:
        gdf = pl.read_csv(out / "fqi_grid.csv")
    else:
        gdf = pl.DataFrame(grid)
        gdf.write_csv(out / "fqi_grid.csv", float_precision=2)
        pl.DataFrame(conv).write_csv(out / "fqi_convergence.csv", float_precision=6)

    def best(rows):
        return rows.sort("val_net_adj_eur", descending=True).row(0, named=True)
    m1 = best(gdf.filter(pl.col("gamma") > 0))
    b3 = best(gdf.filter((pl.col("gamma") == 0) & (pl.col("trees") == m1["trees"])))   # same trees as M1
    choice = {"M1": m1, "B3": b3, "scale_eur_per_day": scale, "selected_on": "validation, seed 0",
              "B3_rule": "gamma = 0 with the tree setting chosen for M1"}
    (out / "fqi_choice.json").write_text(json.dumps(choice, indent=1))
    print(f"chosen M1: {m1}\nchosen B3: {b3}", flush=True)

    # 3. seeds on validation, then refit on training + validation for the test
    seed_rows = []
    for role, c in ((("B3", b3),) if args.only_b3 else (("M1", m1), ("B3", b3))):
        tparams = trees[c["trees"]]
        for seed in SEEDS:
            res = fit_fqi(tr, c["gamma"], c["K"], "extra_trees", tparams, seed=seed, scale=scale)
            qm = res.models[max(res.models)]
            w = fit_projection(tr, qm, "extra_trees", tparams, seed=seed)
            name = {"M1": "M1_learned_value", "B3": "B3_one_step_value"}[role]
            df = val_chain(LearnedValueMILP(fc, w, scale, q0, name=name), f"V2__{name}__validation__seed{seed}.csv")
            seed_rows.append({"method": name, "seed": seed, "val_net_adj_eur": float(df["net_adj_eur"].sum())})
            if role == "M1":
                df2 = val_chain(LearnedEndLevelMILP(fc, qm, q0), f"V2__M2_rl_end_level__validation__seed{seed}.csv")
                seed_rows.append({"method": "M2_rl_end_level", "seed": seed,
                                  "val_net_adj_eur": float(df2["net_adj_eur"].sum())})
            print(f"  {role} seed {seed}: validation {seed_rows[-1]}", flush=True)
            # final model for test / acceptance: same settings, fitted on 2021-2025
            rf = fit_fqi(trva, c["gamma"], c["K"], "extra_trees", tparams, seed=seed, scale=scale)
            qf = rf.models[max(rf.models)]
            wf = fit_projection(trva, qf, "extra_trees", tparams, seed=seed)
            with open(models / f"fqi_{role}_seed{seed}.pkl", "wb") as f:
                pickle.dump({"q_model": qf, "w_model": wf, "scale": scale, "choice": c, "levels": levels.tolist(),
                             "fitted_on": f"{P['train']['start']}..{P['validation']['end']}"}, f)
    seeds_path = out / "fqi_seeds_validation.csv"
    if args.only_b3 and seeds_path.exists():
        seed_rows = pl.read_csv(seeds_path).filter(pl.col("method") != "B3_one_step_value").to_dicts() + seed_rows
    pl.DataFrame(seed_rows).write_csv(seeds_path, float_precision=2)
    print("done", flush=True)


def _d(s):
    import datetime as dt
    return dt.date.fromisoformat(s)


if __name__ == "__main__":
    main()
