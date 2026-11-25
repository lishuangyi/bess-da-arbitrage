"""Summarise the v2 chained backtests and test the paired differences (ablation table).

Usage: python -m bess benchmark v2 --period validation|test|acceptance

Reads <results_dir>/daily/V2__<method>__<period>[__seed<k>].csv (bess backtest chain,
bess train fqi). Daily values are net revenue incl. the terminal valuation (`net_adj_eur`).
Seeded methods: one row per seed and a row with the per-day mean over seeds.
Paired differences M1 - B3, M1 - B0, M1 - B4 (and M1 - B2, M1 - B1, M2 - B3): 7-day moving-block bootstrap
of the daily difference, 5000 resamples, 95 % interval; per seed and for the seed mean.
Profiles with `isea_reference` (configs/v2_isea.yaml) also get the share of the ISEA daily DA index
(configs/isea_daily.csv): <results_dir>/isea_share_<period>.csv.
Writes <results_dir>/summary_<period>.csv and paired_<period>.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from bess.core.config import add_v2_config_arg, v2_config, v2_dir
from bess.input.markets import resolve_market
from bess.output.benchmark import block_bootstrap_means, block_bootstrap_ratio, ci
from bess.output.io import load_daily_seeds, mean_over_seeds
from bess.output.metrics import worst_share_mean

PAIRS = [("M1_learned_value", "B3_one_step_value"), ("M1_learned_value", "B0_fixed50"),
         ("M1_learned_value", "B4_rolling48h"), ("M1_learned_value", "B2_constant_value"),
         ("M1_learned_value", "B1_free_unvalued"),
         ("M2_rl_end_level", "B3_one_step_value")]
ORDER = ["PF_fixed50", "B0_fixed50", "B1_free_unvalued", "B2_constant_value", "B3_one_step_value",
         "B4_rolling48h", "M1_learned_value", "M2_rl_end_level"]


def metrics(df: pl.DataFrame, pf_total: float) -> dict:
    x = df["net_adj_eur"].to_numpy()
    t = df["compute_time_s"].to_numpy()
    return {"n_days": len(x), "net_adj_eur": float(x.sum()), "net_eur": float(df["net_eur"].sum()),
            "terminal_adj_eur": float(df["terminal_adj_eur"].sum()),
            "capture_vs_pf_pct": 100.0 * float(x.sum()) / pf_total,
            "worst5pct_mean_daily_eur": worst_share_mean(x, 0.05), "loss_day_share_pct": 100.0 * float(np.mean(x < 0)),
            "efc_per_day": float(df["efc"].mean()), "violations": int(df["violations"].sum()),
            "time_p95_ms": 1e3 * float(np.quantile(t, 0.95))}


def isea_share(seeded: dict, ref: dict, qh_start: str) -> pl.DataFrame:
    """Share of the ISEA daily DA index (ratio of sums over the same delivery days, EUR per MW).

    `literal`: every day with a published ISEA value, zero-revenue days included;
    `zero_days_dropped`: days where ISEA reports zero revenue removed from both sides.
    Intervals: 7-day moving blocks resampling both sides together, 5000 draws.
    Segments: all days, and the hourly / 15-min parts when both occur.
    """
    from bess.core.config import REPO_ROOT
    path = REPO_ROOT / "configs" / "isea_daily.csv"
    if not path.exists():
        return pl.DataFrame()
    isea = (pl.read_csv(path, try_parse_dates=True)
            .filter((pl.col("system_h") == ref["system_h"]) & (pl.col("cycles") == ref["cycles"]))
            .select(pl.col("day").cast(pl.Date), pl.col("eur_per_mw_day").alias("isea"), "zero"))
    out = []
    for m in sorted(seeded, key=lambda k: ORDER.index(k) if k in ORDER else 99):
        frames = seeded[m]
        seeds = sorted(s for s in frames if s is not None)
        variants = [("mean" if seeds else "-", mean_over_seeds(frames))]
        if len(seeds) > 1:
            variants += [(str(s), frames[s]) for s in seeds]
        for tag, df in variants:
            j = df.select("day", pl.col("net_adj_eur").alias("ours")).join(isea, on="day", how="inner").sort("day")
            for seg, cond in (("all", pl.lit(True)), ("hourly", pl.col("day") < pl.lit(qh_start).str.to_date()),
                              ("15-min", pl.col("day") >= pl.lit(qh_start).str.to_date())):
                js = j.filter(cond)
                if js.height == 0 or (seg != "all" and js.height == j.height):
                    continue
                for variant, jj in (("literal", js), ("zero_days_dropped", js.filter(~pl.col("zero")))):
                    a, b = jj["ours"].to_numpy(), jj["isea"].to_numpy()
                    lo, hi = ci(100.0 * block_bootstrap_ratio(a, b, 7, 5000, 0))
                    out.append({"method": m, "seed": tag, "segment": seg, "variant": variant, "n_days": jj.height,
                                "n_isea_zero_days": int(jj["zero"].sum()), "ours_eur_per_mw": float(a.sum()),
                                "isea_eur_per_mw": float(b.sum()), "share_pct": 100.0 * float(a.sum() / b.sum()),
                                "ci95_lo": lo, "ci95_hi": hi, "first_day": str(jj["day"].min()),
                                "last_day": str(jj["day"].max())})
    return pl.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--period", default="validation")
    add_v2_config_arg(ap)
    args = ap.parse_args(argv)
    v2, base = v2_config(args.config)
    out = v2_dir(v2, "results_dir")
    seeded = {m: frames for (exp, m, per), frames in load_daily_seeds(out / "daily", args.period).items() if exp == "V2"}
    if "PF_fixed50" not in seeded:
        raise SystemExit("PF_fixed50 missing - run bess backtest chain --methods pf first")
    pf_total = float(next(iter(seeded["PF_fixed50"].values()))["net_adj_eur"].sum())
    rows = []
    for m in sorted(seeded, key=lambda k: ORDER.index(k) if k in ORDER else 99):
        frames = seeded[m]
        seeds = sorted(s for s in frames if s is not None)
        rows.append({"method": m, "seed": "mean" if seeds else "-", "n_seeds": len(seeds),
                     **metrics(mean_over_seeds(frames), pf_total)})
        if len(seeds) > 1:
            rows += [{"method": m, "seed": str(s), "n_seeds": 1, **metrics(frames[s], pf_total)} for s in seeds]
    summ = pl.DataFrame(rows)
    summ.write_csv(out / f"summary_{args.period}.csv", float_precision=3)

    pairs = []
    for a, b in PAIRS:
        if a not in seeded or b not in seeded:
            continue
        fa, fb = seeded[a], seeded[b]
        variants = [("mean", mean_over_seeds(fa), mean_over_seeds(fb))]
        common = sorted(set(s for s in fa if s is not None) & set(s for s in fb if s is not None))
        variants += [(str(s), fa[s], fb[s]) for s in common]
        if not common and len([s for s in fa if s is not None]) > 1:   # b deterministic: pair each seed of a with b
            variants += [(str(s), fa[s], mean_over_seeds(fb)) for s in sorted(s for s in fa if s is not None)]
        for tag, da, db in variants:
            j = da.select("day", pl.col("net_adj_eur").alias("a")).join(
                db.select("day", pl.col("net_adj_eur").alias("b")), on="day")
            diff = (j["a"] - j["b"]).to_numpy()
            lo, hi = ci(block_bootstrap_means(diff, 7, 5000, 0))
            pairs.append({"a": a, "b": b, "seed": tag, "n_days": len(diff), "mean_daily_diff_eur": float(diff.mean()),
                          "ci95_lo": lo, "ci95_hi": hi, "total_diff_eur": float(diff.sum()),
                          "significant": not (lo <= 0 <= hi)})
    pdf = pl.DataFrame(pairs) if pairs else pl.DataFrame()
    if pairs:
        pdf.write_csv(out / f"paired_{args.period}.csv", float_precision=3)
    share = pl.DataFrame()
    if v2.get("isea_reference"):
        share = isea_share(seeded, v2["isea_reference"], resolve_market(base).qh_market_start)
        if share.height:
            share.write_csv(out / f"isea_share_{args.period}.csv", float_precision=3)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, float_precision=1, tbl_hide_dataframe_shape=True):
        print(summ)
        if pairs:
            print(pdf)
        if share.height:
            print(share.filter(pl.col("seed").is_in(["-", "mean"])).drop("first_day", "last_day"))


if __name__ == "__main__":
    main()
