"""Build the v2 reward cache: 9 x 9 (start, end SoC) daily MILPs per delivery day, settled at realised prices.

Usage:
  python scripts/build_reward_cache.py --smoke          # one quarter of 2024: timing and an estimate
  python scripts/build_reward_cache.py                  # all days of the cache period
  python scripts/build_reward_cache.py --refit-forecaster

Steps: (1) forward-chaining XGB models per year (models/v2/xgb_forward_fold.pkl, reused if present);
(2) gate-closure forecasts of D and D+1 and the state features phi_D for every evaluable day;
(3) 81 MILPs per day with `n_procs` processes. Writes <cache dir>/reward_cache.parquet,
features.parquet and days.json (evaluable and excluded days; a chain breaks at excluded days).
"""
from __future__ import annotations

import argparse
import json
import time

import polars as pl

from bess.data.smard import load_market_data
from bess.rlvalue.cache import build_cache, features_frame, gate_closure_inputs
from bess.rlvalue.forecasts import ForwardFoldXGB
from bess.rlvalue.periods import add_v2_config_arg, forecaster_path, period_days_res, qh_start, v2_battery, v2_config, v2_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="2024-01-01..2024-03-31 only, written with a _smoke suffix")
    ap.add_argument("--refit-forecaster", action="store_true")
    ap.add_argument("--n-procs", type=int, default=None)
    add_v2_config_arg(ap)
    args = ap.parse_args()
    v2, base = v2_config(args.config)
    gc, pub = base["day_ahead"]["gate_closure_local"], base["day_ahead"]["publication_local"]
    data = load_market_data(base)
    battery = v2_battery(v2)
    models, cache_dir = v2_dir(v2, "models_dir"), v2_dir(v2, "cache")
    models.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    days_res, note = period_days_res(v2, base, data, "cache")
    all_days = [d for d, _ in days_res]
    if args.smoke:
        days_res = [(d, r) for d, r in days_res if str(d) >= "2024-01-01" and str(d) <= "2024-03-31"]
    days = [d for d, _ in days_res]

    fc_path = forecaster_path(v2)          # shared by all profiles: forecasts do not depend on the battery
    fc_path.parent.mkdir(parents=True, exist_ok=True)
    if fc_path.exists() and not args.refit_forecaster:
        fc = ForwardFoldXGB.load(fc_path)
    else:
        t0 = time.perf_counter()
        years = sorted({d.year for d in all_days})      # models for the whole cache period, also in --smoke
        print(f"fitting forward-fold XGB for years {years} ...", flush=True)
        fc = ForwardFoldXGB().fit(data, years)
        fc.save(fc_path)
        print(f"  {len(fc.models)} models in {(time.perf_counter() - t0) / 60:.1f} min; folds: {fc.fold_days}", flush=True)

    t0 = time.perf_counter()
    inputs = gate_closure_inputs(data, fc, days, qh_start(base), gc, pub)
    no_fc = [str(d) for d in days if d not in inputs]
    print(f"gate-closure inputs for {len(inputs)} of {len(days)} days in {time.perf_counter() - t0:.0f} s "
          f"({len(no_fc)} without a forecast model, e.g. the first year)", flush=True)

    n_procs = args.n_procs or v2["cache"]["n_procs"]
    from bess.rlvalue.cache import soc_levels
    n_lv = len(soc_levels(battery))
    print(f"solving {n_lv * n_lv * len(inputs)} MILPs ({n_lv} x {n_lv} levels per day) with {n_procs} processes ...", flush=True)
    t0 = time.perf_counter()
    cache = build_cache(data, inputs, battery, n_procs=n_procs)
    el = time.perf_counter() - t0
    print(f"done in {el / 60:.1f} min ({1e3 * el * n_procs / max(1, cache.height):.1f} ms per MILP and process; "
          f"median solve {cache['solve_ms'].median():.1f} ms); violations: {int(cache['violations'].sum())}", flush=True)

    sfx = "_smoke" if args.smoke else ""
    cache.write_parquet(cache_dir / f"reward_cache{sfx}.parquet")
    features_frame(inputs).write_parquet(cache_dir / f"features{sfx}.parquet")
    (cache_dir / f"days{sfx}.json").write_text(json.dumps({
        "evaluable": [str(d) for d in sorted(inputs)], "no_forecast_model": no_fc,
        "excluded_missing_data": note["excluded_days"], "battery": battery.__dict__}, indent=1, default=str))
    print(f"-> {cache_dir}/reward_cache{sfx}.parquet ({cache.height} rows)")
    if args.smoke:
        full = len(all_days)
        print(f"estimate for the full cache ({full} days): {el / max(1, len(inputs)) * full / 60:.0f} min")


if __name__ == "__main__":
    main()
