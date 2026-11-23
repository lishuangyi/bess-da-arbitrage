"""Download the monthly values of the ISEA Battery Revenue Index into configs/isea_monthly.csv.

Usage: python scripts/fetch_isea_index.py [--months 36] [--from-dir DIR]
       python scripts/fetch_isea_index.py --daily        # daily series -> configs/isea_daily.csv

The public dashboard (https://battery-charts.de/revenue-index/) loads its monthly chart from a
JSON endpoint with a client key that is embedded in the page. This script reads the key from the
page, queries every system (1 h, 2 h) x daily cycle limit (1, 2), and writes one row per month,
system, cycle limit and market. Values are EUR per MW and year (the dashboard's k EUR/MW/a x 1000),
each month annualised. --from-dir converts responses saved earlier as monthly_s<S>_c<C>.json
instead of downloading.

--daily downloads the dashboard's daily "Absolute" series (query avg_revenue_test, avgtype absolute),
day-ahead market only. One request returns at most about 900 days, so windows of 12, 24 and 36 months
are stitched; overlapping days must agree. Values are k EUR/MW/a for the day; the CSV also has
EUR/MW for the day (x 1000 / 365). Days with zero revenue are kept and flagged (`zero`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

from bess.config import REPO_ROOT

PAGE = "https://battery-charts.de/revenue-index/"
API = "https://battery-charts.de/wp-json/isea/v1/data"
SYSTEMS = (1, 2)          # hours of storage at 1 MW (the dashboard's systemtype); 4 h returns no data
CYCLES = (1, 2)
OUT = REPO_ROOT / "configs" / "isea_monthly.csv"
OUT_DAILY = REPO_ROOT / "configs" / "isea_daily.csv"
DAILY_WINDOWS = (12, 24, 36)
UA = {"User-Agent": "bess-backtest/0.1 (research comparison)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def page_key() -> str:
    html = get(PAGE).decode("utf-8", "replace")
    m = re.search(r"api_key=([^&\"'\s]+)", html)
    if not m:
        raise RuntimeError(f"no api_key found on {PAGE}; the dashboard may have changed")
    return urllib.parse.unquote(m.group(1).replace("&amp;", "&"))


def fetch(months: int) -> dict[tuple[int, int], list[dict]]:
    key = page_key()
    out = {}
    for s in SYSTEMS:
        for c in CYCLES:
            q = urllib.parse.urlencode({"api_key": key, "query_id": "monthly_revenues", "systemtype": s,
                                        "cyclelimit": c, "months": months})
            out[s, c] = json.loads(get(f"{API}?{q}"))
            time.sleep(1.0)
    return out


def fetch_daily() -> pl.DataFrame:
    key = page_key()
    rows = {}
    for s in SYSTEMS:
        for c in CYCLES:
            for months in DAILY_WINDOWS:
                q = urllib.parse.urlencode({"api_key": key, "query_id": "avg_revenue_test", "systemtype": s,
                                            "cyclelimit": c, "months": months, "avgtype": "absolute"})
                for r in json.loads(get(f"{API}?{q}")):
                    if r.get("market") != "DA" or r.get("value") is None:
                        continue
                    k = (r["time"][:10], s, c)
                    v = float(r["value"])
                    if k in rows and abs(rows[k] - v) > 1e-9:
                        raise RuntimeError(f"windows disagree on {k}: {rows[k]} vs {v}")
                    rows[k] = v
                time.sleep(1.0)
    today = dt.date.today().isoformat()
    df = pl.DataFrame([{"day": d, "system_h": s, "cycles": c, "k_eur_per_mw_a": v,
                        "eur_per_mw_day": round(1000.0 * v / 365.0, 4), "zero": v == 0.0, "retrieved": today}
                       for (d, s, c), v in rows.items()])
    return df.with_columns(pl.col("day").str.to_date()).sort("system_h", "cycles", "day")


def from_dir(d: Path) -> dict[tuple[int, int], list[dict]]:
    return {(s, c): json.loads((d / f"monthly_s{s}_c{c}.json").read_text()) for s in SYSTEMS for c in CYCLES}


def to_frame(raw: dict[tuple[int, int], list[dict]], retrieved: str) -> pl.DataFrame:
    rows = []
    for (s, c), recs in raw.items():
        for r in recs:
            mm, yyyy = r["month"].split(".")
            for market, v in r.items():
                if market != "month" and v is not None:
                    rows.append({"month": f"{yyyy}-{mm}", "system_h": s, "cycles": c, "market": market,
                                 "eur_per_mw_a": round(1000.0 * float(v), 2), "retrieved": retrieved})
    if not rows:
        raise RuntimeError("the dashboard returned no monthly values")
    return pl.DataFrame(rows).sort("market", "system_h", "cycles", "month")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--months", type=int, default=36, help="history the dashboard is asked for (it has data from 2024)")
    ap.add_argument("--from-dir", default=None, help="convert saved monthly_s<S>_c<C>.json instead of downloading")
    ap.add_argument("--out", default=None)
    ap.add_argument("--daily", action="store_true", help="daily DA series instead of the monthly values")
    args = ap.parse_args()
    if args.daily:
        df = fetch_daily()
        out = args.out or str(OUT_DAILY)
        df.write_csv(out)
        for (s, c), g in df.group_by(["system_h", "cycles"], maintain_order=True):
            print(f"{s} h, {c} cycles: {g.height} days {g['day'].min()}..{g['day'].max()}, {int(g['zero'].sum())} zero days")
        print(f"wrote {out}")
        return
    args.out = args.out or str(OUT)
    raw = from_dir(Path(args.from_dir)) if args.from_dir else fetch(args.months)
    retrieved = (dt.date.fromtimestamp(Path(args.from_dir, "monthly_s2_c2.json").stat().st_mtime).isoformat()
                 if args.from_dir else dt.date.today().isoformat())
    df = to_frame(raw, retrieved)
    df.write_csv(args.out)
    da = df.filter(pl.col("market") == "DA")
    print(f"wrote {args.out}: {df.height} rows, DA {da['month'].min()} to {da['month'].max()}, "
          f"systems {sorted(da['system_h'].unique())} h, cycles {sorted(da['cycles'].unique())}")


if __name__ == "__main__":
    main()
