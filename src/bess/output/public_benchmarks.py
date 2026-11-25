"""Re-run our day-ahead methods under the assumptions of public German BESS benchmarks.

Usage: python -m bess benchmark public [--cases isea,emaps] [--out results/public_benchmarks]

ISEA Battery Revenue Index (RWTH Aachen, with enspired), single-market DA
    Battery and published values come from configs/isea_reference.yaml. One case per year
    (2024, 2025, 2026 to date) x system (1 h, 2 h at 1 MW) x daily cycle limit (1, 2). Prices
    are at the market's own resolution (hourly until 2025-09-30, 15-min after); the case
    isea_2025_2h2c_hourly settles Oct-Dec 2025 at hourly averages instead (sensitivity).
    The index dispatches on realised prices, so the published value is compared with our
    perfect foresight; the forecast-based methods show what a trader committing at gate
    closure keeps. Methods whose forecaster was fitted on a year are left out of that year:
    XGB was trained on 2020-2024 and early-stopped on Jan-Sep 2025 (2025 is not out of sample
    for it; 2026 is).
Electricity Maps x Re-Twin Energy case study (emaps_*)
    German DA, 2026-01-01..03-31, 2 h, 1.5 cycles/day, 95 % per direction, SoC 10-90 %.
    Published: perfect foresight 13,914 EUR/MW, previous-day baseline 10,904, Electricity
    Maps forecast 12,163, Re-Twin forecast 11,628. Its cycle definition and price resolution
    are not stated; the *_hourly and *_usable cases test both.

The daily cycle limit of the ISEA cases counts energy added to the store / nominal capacity per
delivery day, as the index code does (the Electricity Maps cases count discharged energy). Daily
methods start and end every day at 50 % SoC; the rolling 48 h MPC carries SoC over.
Writes <out>/comparison.csv (annual), <out>/monthly.csv and <out>/daily/*.csv.

ISEA monthly values (configs/isea_monthly.csv, from bess isea fetch) are matched
month by month: <out>/isea_monthly.csv (ours and ISEA per month) and <out>/isea_matched.csv
(both day-weighted over the months both cover). --rescore rebuilds these two files from the
existing monthly.csv without running any backtest.
ISEA daily values (configs/isea_daily.csv, bess isea fetch --daily) are matched day by day
with every method: <out>/isea_daily_match.csv (per case and period, literal and with ISEA's
zero-revenue days dropped from both sides) and <out>/isea_daily_2h2c.csv (per day, 2 h / 2 cycles).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import re
from pathlib import Path

import polars as pl

from bess.core.config import (
    REPO_ROOT,
    Battery,
    experiment_config,
    load_yaml,
    models_dir,
    results_dir,
)
from bess.core.markets.day_ahead import DayAheadMarket
from bess.forecasting.naive import SeasonalNaive
from bess.forecasting.xgb import XGBForecaster
from bess.input.markets import resolve_market
from bess.input.periods import segment_days
from bess.input.prices import load_market_data
from bess.milp.policy import ForecastMILP, PerfectForesight
from bess.milp.rolling import RollingMILP
from bess.output.backtest.engine import run_backtest
from bess.output.metrics import summarize

QH_START = "2025-10-01"   # first native 15-min DA delivery day
ISEA_MONTHLY = REPO_ROOT / "configs" / "isea_monthly.csv"
ISEA_DAILY = REPO_ROOT / "configs" / "isea_daily.csv"
CASE_RE = r"^isea_(\d{4})_(\d)h(\d)c(_hourly)?$"


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    family: str              # "isea" or "emaps"
    battery: Battery
    segments: tuple          # ((start, end, resolution), ...)
    methods: tuple           # method names to run (see build_policies)
    annualize: bool          # EUR/MW/a (True) or EUR/MW over the window (False)
    published: dict          # method -> published value in the same unit
    meta: dict               # year, system_h, cycles, ...
    source: str = ""


def build_policies(xgb) -> dict:
    d1 = SeasonalNaive(lag_days=1)
    d1.name = "naive_d1"     # "previous-day" baseline: prices of D-1, known at gate closure
    pols = [PerfectForesight(), ForecastMILP(SeasonalNaive()), ForecastMILP(d1)]
    if xgb is not None:
        pols += [ForecastMILP(xgb), RollingMILP(xgb)]
    return {p.name: p for p in pols}


ALL = ("perfect_foresight", "forecast_milp_naive", "forecast_milp_naive_d1", "forecast_milp_xgb",
       "rolling48h_milp_xgb")
NO_XGB = ALL[:3]


def native_segments(start: str, end: str) -> tuple:
    """Split [start, end] at the 15-min switch: hourly before, quarter-hour from QH_START on."""
    segs = []
    if start < QH_START:
        segs.append((start, min(end, "2025-09-30"), "hour"))
    if end == "latest" or end >= QH_START:
        segs.append((max(start, QH_START), end, "quarterhour"))
    return tuple(segs)


def isea_cases(ref: dict) -> list[Case]:
    a = ref["assumptions"]
    eta = a["round_trip_efficiency"] ** 0.5
    published = {(r["year"], r["system_h"], r["cycles"]): r for r in ref["annual"]}
    years = {2024: ("2024-01-01", "2024-12-31", NO_XGB),
             2025: ("2025-01-01", "2025-12-31", ALL),
             2026: ("2026-01-01", "latest", ALL)}
    out = []
    for year, (start, end, methods) in years.items():
        for system_h in (1, 2):
            for cycles in (1, 2):
                b = Battery(p_max_mw=a["p_max_mw"], e_max_mwh=a["p_max_mw"] * system_h, eta_ch=eta, eta_dis=eta,
                            soc_min=a["soc_min"], soc_max=a["soc_max"], soc_init=0.5, soc_final=0.5,
                            c_deg_eur_per_mwh=0.0, max_cycles_per_day=float(cycles), cycle_basis="stored_charge")
                pub = published.get((year, system_h, cycles))
                meta = {"year": year, "system_h": system_h, "cycles": cycles, "prices": "native"}
                name = f"isea_{year}_{system_h}h{cycles}c"
                case = Case(name, "isea", b, native_segments(start, end), methods, True,
                            {"perfect_foresight": pub["eur_per_mw_a"]} if pub else {}, meta,
                            pub["source"] if pub else ref["index"]["dashboard"])
                out.append(case)
                if (year, system_h, cycles) == (2025, 2, 2):
                    out.append(dataclasses.replace(case, name=name + "_hourly",
                                                   segments=((start, end, "hour"),),
                                                   meta={**meta, "prices": "hourly"}))
    return out


def emaps_cases() -> list[Case]:
    b = Battery(p_max_mw=1.0, e_max_mwh=2.0, eta_ch=0.95, eta_dis=0.95, soc_min=0.1, soc_max=0.9,
                soc_init=0.5, soc_final=0.5, c_deg_eur_per_mwh=0.0, max_cycles_per_day=1.5)
    pub = {"perfect_foresight": 13914.0, "forecast_milp_naive_d1": 10904.0,
           "(Electricity Maps forecast)": 12163.0, "(Re-Twin forecast)": 11628.0}
    src = ("https://www.electricitymaps.com/resources/case-studies/"
           "re-twin-energy-measures-how-much-a-better-day-ahead-forecast-is-worth-to-a-battery")
    base = Case("emaps_q1", "emaps", b, (("2026-01-01", "2026-03-31", "quarterhour"),),
                ("perfect_foresight", "forecast_milp_naive", "forecast_milp_naive_d1", "forecast_milp_xgb"),
                False, pub, {"year": 2026, "system_h": 2, "cycles": 1.5, "prices": "native"}, src)
    usable = dataclasses.replace(b, max_cycles_per_day=1.2)   # 1.5 cycles of the usable 1.6 MWh
    hourly = (("2026-01-01", "2026-03-31", "hour"),)
    return [base,
            dataclasses.replace(base, name="emaps_q1_usable", battery=usable),
            dataclasses.replace(base, name="emaps_q1_hourly", segments=hourly, meta={**base.meta, "prices": "hourly"}),
            dataclasses.replace(base, name="emaps_q1_hourly_usable", battery=usable, segments=hourly,
                                meta={**base.meta, "prices": "hourly"})]


def monthly_rows(case: Case, method: str, daily: pl.DataFrame) -> list[dict]:
    m = (daily.with_columns(pl.col("day").cast(pl.Date).dt.strftime("%Y-%m").alias("month"))
         .group_by("month").agg(pl.col("net_eur").sum().alias("net"), pl.len().alias("n"),
                                pl.col("efc").mean().alias("efc"))
         .sort("month"))
    return [{"case": case.name, "method": method, "month": r["month"], "n_days": r["n"],
             "net_eur": r["net"], "eur_per_mw_a": r["net"] / r["n"] * 365.0 / case.battery.p_max_mw,
             "efc_per_day": r["efc"]} for r in m.iter_rows(named=True)]


def merge_write(new: pl.DataFrame, path: Path) -> None:
    """Replace the rows of the cases in `new`, keep the others."""
    if path.exists():
        old = pl.read_csv(path, infer_schema_length=0)
        old = old.filter(~pl.col("case").is_in(new["case"].unique().to_list()))
        new = pl.concat([old, new.cast({c: pl.String for c in new.columns})], how="diagonal_relaxed")
    new.write_csv(path, float_precision=2)


def isea_match(out: Path) -> None:
    """Match our monthly values with the ISEA dashboard's monthly DA values (same months only)."""
    if not ISEA_MONTHLY.exists() or not (out / "monthly.csv").exists():
        print(f"skip ISEA matching: {ISEA_MONTHLY.name} or monthly.csv missing (run bess isea fetch)")
        return
    isea = (pl.read_csv(ISEA_MONTHLY).filter(pl.col("market") == "DA")
            .select("month", "system_h", "cycles", pl.col("eur_per_mw_a").alias("isea")))
    m = (pl.read_csv(out / "monthly.csv", schema_overrides={"month": pl.String})
         .filter(pl.col("case").str.contains(CASE_RE))
         .with_columns(pl.col("case").str.extract(CASE_RE, 1).cast(pl.Int64).alias("year"),
                       pl.col("case").str.extract(CASE_RE, 2).cast(pl.Int64).alias("system_h"),
                       pl.col("case").str.extract(CASE_RE, 3).cast(pl.Int64).alias("cycles"),
                       pl.when(pl.col("case").str.ends_with("_hourly")).then(pl.lit("hourly"))
                       .otherwise(pl.lit("native")).alias("prices"))
         .join(isea, on=["month", "system_h", "cycles"], how="inner")
         .with_columns(pl.col("month").str.slice(0, 7).str.to_date("%Y-%m")
                       .dt.month_end().dt.day().alias("days_in_month")))
    # only whole months on both sides: a month we cover partly is not comparable with ISEA's month
    m = m.filter(pl.col("n_days") == pl.col("days_in_month"))
    keys = ["case", "method", "year", "system_h", "cycles", "prices"]
    monthly = (m.select(*keys, "month", "n_days", pl.col("eur_per_mw_a").alias("ours"), "isea")
               .with_columns((100.0 * pl.col("ours") / pl.col("isea")).alias("ours_vs_isea_pct"))
               .sort(keys + ["month"]))
    monthly.write_csv(out / "isea_monthly.csv", float_precision=2)
    w = pl.col("n_days")
    matched = (monthly.group_by(keys)
               .agg(pl.len().alias("months"), pl.col("month").min().alias("first_month"),
                    pl.col("month").max().alias("last_month"), w.sum().alias("n_days"),
                    ((pl.col("ours") * w).sum() / w.sum()).alias("ours"),
                    ((pl.col("isea") * w).sum() / w.sum()).alias("isea"),
                    pl.col("ours_vs_isea_pct").min().alias("month_pct_min"),
                    pl.col("ours_vs_isea_pct").max().alias("month_pct_max"))
               .with_columns((100.0 * pl.col("ours") / pl.col("isea")).alias("ours_vs_isea_pct"))
               .sort(keys))
    matched.write_csv(out / "isea_matched.csv", float_precision=2)
    print(f"-> {out / 'isea_matched.csv'} ({matched.height} rows), {out / 'isea_monthly.csv'} ({monthly.height} rows)")
    isea_daily_match(out)


def isea_daily_match(out: Path) -> None:
    """Day-by-day match of every method (perfect foresight and the gate-closure methods) with the ISEA daily DA series."""
    if not ISEA_DAILY.exists():
        print(f"skip daily ISEA matching: {ISEA_DAILY.name} missing (bess isea fetch --daily)")
        return
    isea = pl.read_csv(ISEA_DAILY, try_parse_dates=True).select(
        pl.col("day").cast(pl.Date), "system_h", "cycles", pl.col("eur_per_mw_day").alias("isea"), "zero")
    rows, per_day = [], []
    for f in sorted((out / "daily").glob("isea_*__*.csv")):
        case, method = f.stem.split("__")
        m = re.match(CASE_RE, case)
        if not m:
            continue
        sys_h, cyc, prices = int(m.group(2)), int(m.group(3)), "hourly" if m.group(4) else "native"
        d = pl.read_csv(f, try_parse_dates=True).select(pl.col("day").cast(pl.Date), pl.col("net_eur").alias("ours"))
        j = d.join(isea.filter((pl.col("system_h") == sys_h) & (pl.col("cycles") == cyc)), on="day").sort("day")
        if j.height == 0:
            continue
        if (sys_h, cyc, prices) == (2, 2, "native"):
            per_day.append(j.with_columns(pl.lit(case).alias("case"), pl.lit(method).alias("method")))
        for variant, jj in (("literal", j), ("zero_days_dropped", j.filter(~pl.col("zero")))):
            rows.append({"case": case, "method": method, "year": int(m.group(1)), "system_h": sys_h, "cycles": cyc, "prices": prices, "variant": variant,
                         "first_day": str(jj["day"].min()), "last_day": str(jj["day"].max()), "n_days": jj.height,
                         "n_isea_zero_days": int(j["zero"].sum()) if variant == "literal" else 0,
                         "ours_eur_per_mw": float(jj["ours"].sum()), "isea_eur_per_mw": float(jj["isea"].sum()),
                         "ours_vs_isea_pct": 100.0 * float(jj["ours"].sum() / jj["isea"].sum()),
                         "median_daily_ratio_pct": 100.0 * float((jj.filter(pl.col("isea") > 0)["ours"]
                                                                  / jj.filter(pl.col("isea") > 0)["isea"]).median())})
    pl.DataFrame(rows).write_csv(out / "isea_daily_match.csv", float_precision=3)
    if per_day:
        pl.concat(per_day).write_csv(out / "isea_daily_2h2c.csv", float_precision=3)
    print(f"-> {out / 'isea_daily_match.csv'} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="isea,emaps",
                    help="comma-separated families (isea, emaps) and/or case names")
    ap.add_argument("--out", default=None, help="output directory, default results/public_benchmarks")
    ap.add_argument("--rescore", action="store_true",
                    help="only rebuild the ISEA month-by-month match from the existing monthly.csv")
    args = ap.parse_args(argv)
    cfg = experiment_config()
    out = Path(args.out) if args.out else results_dir(cfg) / "public_benchmarks"
    if not out.is_absolute():
        out = REPO_ROOT / out
    if args.rescore:
        isea_match(out)
        return
    spec = resolve_market(cfg)
    gc, pub_hhmm = spec.gate_closure_local, spec.publication_local
    market = DayAheadMarket(gc)
    data = load_market_data(cfg)
    xgb_path = models_dir(cfg) / "xgb.pkl"
    pols = build_policies(XGBForecaster.load(xgb_path) if xgb_path.exists() else None)
    ref = load_yaml("isea_reference.yaml")
    cases = isea_cases(ref) + emaps_cases()
    want = set(args.cases.split(","))
    cases = [c for c in cases if c.family in want or c.name in want]
    (out / "daily").mkdir(parents=True, exist_ok=True)
    rows, months = [], []
    for case in cases:
        dailies, excluded = {}, 0
        for mname in case.methods:
            if mname not in pols:
                continue
            parts = []
            for start, end, res in case.segments:
                days, res, info = segment_days(data, start, end, res)
                if mname == "perfect_foresight":
                    excluded += info["n_incomplete_excluded"] + info["n_missing_inputs_excluded"]
                if days:
                    parts.append(run_backtest(pols[mname], data, days, res, case.battery, market,
                                              publication_hhmm=pub_hhmm))
            daily = pl.concat(parts)
            daily.write_csv(out / "daily" / f"{case.name}__{mname}.csv")
            dailies[mname] = daily
            months += monthly_rows(case, mname, daily)
        ref_daily = dailies["perfect_foresight"]
        first, last = str(ref_daily["day"].min()), str(ref_daily["day"].max())
        for mname, daily in dailies.items():
            s = summarize(daily, ref_daily)
            n = s["n_days"]
            value = s["net_revenue_eur"] / case.battery.p_max_mw * (365.0 / n if case.annualize else 1.0)
            pubv = case.published.get(mname)
            rows.append({**case.meta, "case": case.name, "family": case.family, "method": mname,
                         "first_day": first, "last_day": last, "n_days": n, "days_excluded": excluded,
                         "unit": "EUR/MW/a" if case.annualize else "EUR/MW (window)",
                         "ours": value, "published": pubv,
                         "ours_vs_published_pct": None if pubv is None else 100.0 * value / pubv,
                         "capture_rate_pct": s["capture_rate_pct"], "efc_per_day": s["efc_per_day"],
                         "violations": s["violations"], "source": case.source})
            print(f"{case.name:24s} {mname:24s} {value:12,.0f}  capture={s['capture_rate_pct']:6.2f}%  "
                  f"efc/day={s['efc_per_day']:.3f}  viol={s['violations']}"
                  + ("" if pubv is None else f"  published={pubv:,.0f}"))
        for k, v in case.published.items():
            if k.startswith("("):   # published methods we cannot run (vendor forecasts)
                rows.append({**case.meta, "case": case.name, "family": case.family, "method": k,
                             "unit": "EUR/MW (window)", "published": v, "source": case.source,
                             "capture_rate_pct": 100.0 * v / case.published["perfect_foresight"]})
    merge_write(pl.DataFrame(rows, infer_schema_length=None), out / "comparison.csv")
    merge_write(pl.DataFrame(months), out / "monthly.csv")
    isea_match(out)
    print(f"-> {out / 'comparison.csv'}, {out / 'monthly.csv'} ({dt.datetime.now():%H:%M})")


if __name__ == "__main__":
    main()
