"""Chained backtests of the v2 baselines (end-of-day SoC carried from day to day).

Usage:
  python -m bess backtest chain --period validation --methods pf,b0,b1,b2,b4 [--b2-grid 0,10,...]
  python -m bess backtest chain --period test --methods pf,b0,b1,b2,b4      # B2 uses the constant chosen on validation

Methods (all on the forward-chaining XGB forecasts, models/v2/xgb_forward_fold.pkl):
  pf  perfect foresight, end of day fixed at 50 % (reference; an upper bound only for fixed-50 % strategies)
  b0  daily MILP, end of day fixed at 50 % (v1 E2 with forward-fold forecasts)
  b1  daily MILP, end of day free and not valued (lower reference: tends to empty the battery)
  b2  daily MILP, end-of-day energy valued at a constant EUR/MWh, tuned on the validation period
  b4  rolling 48 h MILP (v1 E3 with forward-fold forecasts)
  b3, m1, m2  learned-value policies (test / acceptance; models from bess train fqi, seeds 0-2):
      b3 = MILP + one-step value (gamma = 0), m1 = MILP + learned multi-step value, m2 = Q picks the end level
Writes <results_dir>/daily/V2__<method>__<period>.csv, <results_dir>/b2_grid.csv (validation) and
<results_dir>/b2_choice.json.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import polars as pl

from bess.core.config import (
    add_v2_config_arg,
    forecaster_path,
    v2_battery,
    v2_config,
    v2_dir,
)
from bess.forecasting.forward_fold import ForwardFoldXGB
from bess.input.markets import resolve_market
from bess.input.periods import period_days_res
from bess.input.prices import load_market_data
from bess.milp.policy import ForecastMILP, PerfectForesight
from bess.milp.rolling import RollingMILP
from bess.output.backtest.chain import mean_period_price, run_chain
from bess.rl.value.policy import (
    LearnedEndLevelMILP,
    LearnedValueMILP,
    LinearResidualMILP,
)

NAMES = {"pf": "PF_fixed50", "b0": "B0_fixed50", "b1": "B1_free_unvalued", "b2": "B2_constant_value",
         "b4": "B4_rolling48h", "b3": "B3_one_step_value", "m1": "M1_learned_value", "m2": "M2_rl_end_level"}
SEEDS = (0, 1, 2)
DEFAULT_GRID = "0,10,20,30,40,50,60,70,80,90,100,120,150"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--period", default="validation", choices=["validation", "test", "acceptance"])
    ap.add_argument("--methods", default="pf,b0,b1,b2,b4")
    ap.add_argument("--b2-grid", default=DEFAULT_GRID, help="EUR/MWh values tried on validation")
    add_v2_config_arg(ap)
    args = ap.parse_args(argv)
    v2, base = v2_config(args.config)
    spec = resolve_market(base)
    pub = spec.publication_local
    market = spec.day_ahead_market()
    data = load_market_data(base)
    battery = v2_battery(v2)
    models_dir = v2_dir(v2, "models_dir")
    fc = ForwardFoldXGB.load(forecaster_path(v2))
    q0 = spec.qh_start_date
    out = v2_dir(v2, "results_dir")
    (out / "daily").mkdir(parents=True, exist_ok=True)
    days_res, note = period_days_res(v2, base, data, args.period)
    p_bar = mean_period_price(data, days_res)
    print(f"{args.period}: {len(days_res)} days {days_res[0][0]}..{days_res[-1][0]}, "
          f"{len(note['excluded_days'])} excluded, mean price {p_bar:.2f} EUR/MWh", flush=True)

    def run(policy, key, display=None):
        df = run_chain(policy, data, days_res, battery, market, pub, mean_price=p_bar)
        fname = f"V2__{key}.csv" if "__" in key else f"V2__{key}__{args.period}.csv"
        df.write_csv(out / "daily" / fname)
        key = display or key
        print(f"  {key:22s} net={df['net_eur'].sum():10.0f}  +terminal={df['terminal_adj_eur'].sum():7.0f}  "
              f"efc/day={df['efc'].mean():.2f}  viol={int(df['violations'].sum())}", flush=True)
        return df

    choice_path = out / "b2_choice.json"
    for m in args.methods.split(","):
        if m == "pf":
            run(PerfectForesight(), NAMES[m])
        elif m == "b0":
            run(ForecastMILP(fc, name=NAMES[m]), NAMES[m])
        elif m == "b1":
            run(LinearResidualMILP(fc, 0.0, name=NAMES[m]), NAMES[m])
        elif m == "b4":
            run(RollingMILP(fc, name=NAMES[m]), NAMES[m])
        elif m == "b2":
            if args.period == "validation":
                rows = []
                for c in [float(x) for x in args.b2_grid.split(",")]:
                    df = run_chain(LinearResidualMILP(fc, c), data, days_res, battery, market, pub, mean_price=p_bar)
                    rows.append({"eur_per_mwh": c, "net_adj_eur": float(df["net_adj_eur"].sum()),
                                 "net_eur": float(df["net_eur"].sum()), "efc_per_day": float(df["efc"].mean())})
                    print(f"  B2 grid {c:6.1f} EUR/MWh: net+terminal={rows[-1]['net_adj_eur']:10.0f}", flush=True)
                grid = pl.DataFrame(rows)
                grid.write_csv(out / "b2_grid.csv", float_precision=2)
                best = float(grid["eur_per_mwh"][int(np.argmax(grid["net_adj_eur"].to_numpy()))])
                choice_path.write_text(json.dumps({"eur_per_mwh": best, "chosen_on": "validation"}))
                print(f"  B2: chose {best:g} EUR/MWh on validation", flush=True)
            best = json.loads(choice_path.read_text())["eur_per_mwh"]
            run(LinearResidualMILP(fc, best, name=NAMES[m]), NAMES[m])
        elif m in ("m1", "b3", "m2"):
            if args.period == "validation":
                raise SystemExit("m1/b3/m2 on validation are run by bess train fqi (models fitted on 2021-2024)")
            role = "B3" if m == "b3" else "M1"
            for seed in SEEDS:
                with open(models_dir / f"fqi_{role}_seed{seed}.pkl", "rb") as f:
                    fm = pickle.load(f)
                if m == "m2":
                    pol = LearnedEndLevelMILP(fc, fm["q_model"], q0, name=NAMES[m])
                else:
                    pol = LearnedValueMILP(fc, fm["w_model"], fm["scale"], q0, name=NAMES[m])
                run(pol, f"{NAMES[m]}__{args.period}__seed{seed}", display=f"{NAMES[m]} seed {seed}")
        else:
            raise SystemExit(f"unknown method {m}")


if __name__ == "__main__":
    main()
