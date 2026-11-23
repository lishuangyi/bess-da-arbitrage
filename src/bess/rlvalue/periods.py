"""v2 periods: delivery days of a period at their native resolution (hourly before the 15-min switch)."""
from __future__ import annotations

import datetime as dt

from bess.backtest.periods import segment_days
from bess.config import REPO_ROOT, experiment_config, load_yaml
from bess.data import calendar as cal
from bess.data.smard import MarketData


def v2_config(name: str = "v2.yaml") -> tuple[dict, dict]:
    """(v2 config, base experiment config)."""
    v2 = load_yaml(name)
    return v2, experiment_config(v2["base_config"])


def v2_battery(v2: dict):
    from bess.config import Battery
    return Battery.from_yaml(v2.get("battery", "battery.yaml"))


def forecaster_path(v2: dict):
    return REPO_ROOT / v2.get("forecaster", "models/v2/xgb_forward_fold.pkl")


def add_v2_config_arg(ap) -> None:
    ap.add_argument("--config", default="v2.yaml",
                    help="v2 profile in configs/: v2.yaml (project battery) or v2_isea.yaml (ISEA index battery)")


def qh_start(base_cfg: dict) -> dt.date:
    return cal.to_date(base_cfg["data"]["qh_market_start"])


def period_days_res(v2: dict, base_cfg: dict, data: MarketData, period: str) -> tuple[list[tuple], dict]:
    """[(day, resolution)] of the evaluable days of a v2 period, and a note with excluded days."""
    p = v2["periods"][period]
    q = qh_start(base_cfg)
    start = cal.to_date(p["start"])
    end = data.quarterhour.days()[-1] if p["end"] == "latest" else cal.to_date(p["end"])
    out, note = [], {"start": str(start), "end": str(end), "excluded_days": []}
    segs = []
    if start < q:
        segs.append((start, min(end, q - dt.timedelta(days=1)), "hour"))
    if end >= q:
        segs.append((max(start, q), end, "quarterhour"))
    for a, b, res in segs:
        if a > b:
            continue
        days, res, info = segment_days(data, a, b, res)
        out += [(d, res) for d in days]
        note["excluded_days"] += info["excluded_days"]
    return out, note


def v2_dir(v2: dict, key: str):
    return REPO_ROOT / (v2[key] if key != "cache" else v2["cache"]["dir"])
