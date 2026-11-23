"""Backtest metrics (computed from per-day results)."""
from __future__ import annotations

import math

import numpy as np
import polars as pl


def worst_share_mean(x: np.ndarray, share: float = 0.05) -> float:
    k = max(1, math.ceil(share * len(x)))
    return float(np.sort(x)[:k].mean())


def summarize(daily: pl.DataFrame, reference: pl.DataFrame | None = None) -> dict:
    """Aggregate metrics. `reference` = perfect-foresight daily results on the same days."""
    net = daily["net_eur"].to_numpy()
    t = daily["compute_time_s"].to_numpy()
    out = {
        "n_days": int(daily.height),
        "net_revenue_eur": float(net.sum()),
        "mean_daily_net_eur": float(net.mean()),
        "capture_rate_pct": float("nan"),
        "worst5pct_mean_daily_eur": worst_share_mean(net, 0.05),
        "loss_day_share_pct": float(100.0 * np.mean(net < 0)),
        "efc_per_day": float(daily["efc"].mean()),
        "violations": int(daily["violations"].sum()),
        "time_p50_ms": float(1e3 * np.quantile(t, 0.5)),
        "time_p95_ms": float(1e3 * np.quantile(t, 0.95)),
        # share of steps whose RL action the feasibility projection changed (RL policies only)
        "action_clip_pct": (float(100.0 * daily["action_clip_share"].mean())
                            if "action_clip_share" in daily.columns else float("nan")),
    }
    if reference is not None:
        ref = dict(zip(reference["day"].to_list(), reference["net_eur"].to_list()))
        missing = [d for d in daily["day"].to_list() if d not in ref]
        if missing:
            raise ValueError(f"reference missing {len(missing)} days")
        out["capture_rate_pct"] = float(100.0 * net.sum() / sum(ref[d] for d in daily["day"].to_list()))
    return out
