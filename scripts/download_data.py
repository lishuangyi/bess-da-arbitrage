"""Download SMARD DE-LU day-ahead prices (hour + quarterhour) and cache them as parquet.

Data: Bundesnetzagentur | SMARD.de (CC BY 4.0). Raw data is not redistributed.

Usage: python scripts/download_data.py [--cache-dir data] [--config experiment.yaml]
Writes data/prices_hour.parquet, data/prices_quarterhour.parquet and
results/data_report.json (missing values, DST step counts, duplication checks).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bess.config import REPO_ROOT, experiment_config
from bess.data import smard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--config", default="experiment.yaml", help="base config (another SMARD price series, e.g. experiment_at.yaml)")
    args = ap.parse_args()
    cfg = experiment_config(args.config)
    cache = Path(args.cache_dir or REPO_ROOT / cfg["data"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)

    series = {}
    for res, dt_h in (("hour", 1.0), ("quarterhour", 0.25)):
        paths = smard.download_raw(cfg, cache, res)
        s = smard.parse_raw(paths, dt_h).starting_at(smard.cal.day_start_utc(cfg["data"]["download_start"]))
        out = smard.cache_paths(cache)[res]
        s.to_frame().write_parquet(out)
        series[res] = s
        print(f"{res}: {len(s.price)} rows, {int(np.isnan(s.price).sum())} missing, "
              f"{s.first_ts} .. {s.last_ts} -> {out}")

    rep = smard.quality_report(series["hour"], series["quarterhour"], cfg["data"]["qh_market_start"])
    res_dir = REPO_ROOT / cfg["backtest"]["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "data_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        if isinstance(v, dict) and "incomplete_days" in v:
            v = {**v, "incomplete_days": f"{len(v['incomplete_days'])} days: {v['incomplete_days'][:10]}"}
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
