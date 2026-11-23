"""Reading per-day result files.

Deterministic methods: results/daily/<experiment>__<method>__<period>.csv
Seeded methods (RL):   results/daily/<experiment>__<method>__<period>__seed<k>.csv
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

# per-day columns averaged over seeds; violations are summed so that any violation stays visible
_MEAN_COLS = ["revenue_eur", "deg_cost_eur", "net_eur", "discharge_mwh", "charge_mwh", "efc", "e_start_mwh",
              "e_end_mwh", "compute_time_s", "wall_time_s", "action_clip_share",
              "terminal_adj_eur", "net_adj_eur", "end_level", "terminal_value_eur"]


def read_daily(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True).with_columns(pl.col("day").cast(pl.Date))


def _parse(stem: str) -> tuple[str, str, str, int | None]:
    parts = stem.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], None
    if len(parts) == 4 and parts[3].startswith("seed"):
        return parts[0], parts[1], parts[2], int(parts[3][4:])
    raise ValueError(f"unexpected daily result file name: {stem}")


def load_daily_seeds(daily_dir: Path, period: str | None = None) -> dict[tuple[str, str, str], dict[int | None, pl.DataFrame]]:
    """{(experiment, method, period): {seed or None: per-day frame}}."""
    out: dict = defaultdict(dict)
    for f in sorted(Path(daily_dir).glob("*__*__*.csv")):
        exp, method, per, seed = _parse(f.stem)
        if period is None or per == period:
            out[(exp, method, per)][seed] = read_daily(f)
    return dict(out)


def mean_over_seeds(frames: dict[int | None, pl.DataFrame]) -> pl.DataFrame:
    """Per-day mean over seeds (violations summed). A single frame is returned unchanged."""
    if len(frames) == 1:
        return next(iter(frames.values()))
    dfs = list(frames.values())
    days = dfs[0]["day"].to_list()
    if any(df["day"].to_list() != days for df in dfs[1:]):
        raise ValueError("seeds were evaluated on different days")
    stacked = pl.concat(dfs, how="diagonal_relaxed")
    aggs = [pl.col(c).mean() for c in _MEAN_COLS if c in stacked.columns]
    aggs += [pl.col("violations").sum(), pl.col("n_steps").first()]
    return stacked.group_by("day", maintain_order=True).agg(aggs)


def load_daily_dir(daily_dir: Path, period: str | None = None) -> dict[tuple[str, str, str], pl.DataFrame]:
    """{(experiment, method, period): per-day frame}; seeded methods are averaged over their seeds."""
    return {k: mean_over_seeds(v) for k, v in load_daily_seeds(daily_dir, period).items()}
