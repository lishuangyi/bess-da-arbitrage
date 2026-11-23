"""Evaluation periods (data splits) from the experiment config.

A split is either one segment ``{start, end, resolution}`` or a list of such
segments (e.g. a training set made of hourly days followed by 15-min days).
"""
from __future__ import annotations

import datetime as dt

from bess.data import calendar as cal
from bess.data.smard import MarketData

# Hourly history every forecaster may need at gate closure of D (XGB lags D-1, D-2, D-7).
REQUIRED_INPUT_LAGS = (1, 2, 7)


def _segment_days(data: MarketData, seg: dict) -> tuple[list, str, dict]:
    res = seg["resolution"]
    series = data.series(res)
    end = series.days()[-1] if seg["end"] == "latest" else cal.to_date(seg["end"])
    all_days = cal.daterange(seg["start"], end)
    complete = [d for d in all_days if series.is_complete(d)]
    days = [d for d in complete
            if all(data.hour.is_complete(d - dt.timedelta(days=k)) for k in REQUIRED_INPUT_LAGS)]
    info = {"start": str(all_days[0]), "end": str(all_days[-1]), "n_calendar_days": len(all_days),
            "n_incomplete_excluded": len(all_days) - len(complete),
            "n_missing_inputs_excluded": len(complete) - len(days),
            "excluded_days": [str(d) for d in all_days if d not in set(days)]}
    return days, res, info


def segment_days(data: MarketData, start, end, resolution: str) -> tuple[list, str, dict]:
    """Evaluable days of an ad-hoc segment (same filtering rules as the config splits)."""
    return _segment_days(data, {"start": str(start), "end": str(end), "resolution": resolution})


def split_segments(cfg: dict, data: MarketData, period: str) -> list[tuple[list, str, dict]]:
    """[(evaluable days, resolution, data note)] for every segment of a split.

    A day is evaluable if (a) its realised prices are complete and (b) the hourly
    prices of D-1, D-2 and D-7 (gate-closure inputs) are complete. Days failing
    (b) are excluded for *all* methods so that every method is evaluated on the
    same days; nothing is interpolated.
    """
    sp = cfg["splits"][period]
    return [_segment_days(data, seg) for seg in (sp if isinstance(sp, list) else [sp])]


def period_days(cfg: dict, data: MarketData, period: str) -> tuple[list, str, dict]:
    """Days, resolution and data note of a single-segment split (validation and test)."""
    segs = split_segments(cfg, data, period)
    if len(segs) != 1:
        raise ValueError(f"split '{period}' has {len(segs)} segments; use split_segments()")
    return segs[0]


def all_split_days(cfg: dict, data: MarketData, period: str) -> list:
    """Sorted evaluable days of a split over all its segments (any resolution)."""
    return sorted(d for days, _, _ in split_segments(cfg, data, period) for d in days)
