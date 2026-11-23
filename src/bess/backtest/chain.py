"""Chained backtest (v2): methods run day after day and carry their end-of-day SoC.

Rules (identical for every method):
- the first day of a period starts at 50 % SoC;
- a chain breaks at a missing delivery day (a day that cannot be evaluated); every method
  then restarts at 50 % on the next evaluable day;
- terminal valuation: at the end of every chain (before a break and at the end of the period)
  the energy left above (or below) the 50 % restart level is valued at the mean realised price
  of the period times the discharge efficiency:
      terminal_adj_eur = (e_end - e_restart) * mean_price * eta_dis
  It is booked on the chain's last day; `net_adj_eur` = `net_eur` + `terminal_adj_eur`.
  Methods that end every day at 50 % get 0.
Timing, information set and settlement are those of the daily backtester (bess.backtest.engine).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from bess.backtest.engine import settle_day
from bess.backtest.policy import Policy
from bess.config import Battery
from bess.data import calendar as cal
from bess.data.smard import MarketData
from bess.markets.base import Market


def mean_period_price(data: MarketData, days_res: list[tuple]) -> float:
    """Time-weighted mean realised price over the period's days (native resolution)."""
    num = den = 0.0
    for d, res in days_res:
        p = data.series(res).day_prices(d)
        num += float(np.sum(p)) * cal.RESOLUTION_HOURS[res]
        den += len(p) * cal.RESOLUTION_HOURS[res]
    return num / den


def chain_breaks(days: list) -> list[bool]:
    """True where a day does not follow the previous evaluable day (chain restarts)."""
    return [i == 0 or days[i] - days[i - 1] != dt.timedelta(days=1) for i in range(len(days))]


def run_chain(policy: Policy, data: MarketData, days_res: list[tuple], battery: Battery, market: Market,
              publication_hhmm: str = "13:00", mean_price: float | None = None) -> pl.DataFrame:
    """`days_res`: [(day, resolution)] sorted by day (one period, possibly mixed resolution)."""
    days = [cal.to_date(d) for d, _ in days_res]
    if days != sorted(days):
        raise ValueError("days must be sorted")
    p_bar = mean_period_price(data, days_res) if mean_price is None else mean_price
    restart = chain_breaks(days)
    rows, e_init = [], battery.e_init
    for i, (d, res) in enumerate(days_res):
        if restart[i]:
            e_init = battery.e_init
        row = settle_day(policy, data, d, res, e_init, battery, market, publication_hhmm)
        row["chain_start"] = restart[i]
        rows.append(row)
        e_init = row["e_end_mwh"]
    df = pl.DataFrame(rows)
    chain_end = [restart[i + 1] if i + 1 < len(days) else True for i in range(len(days))]
    adj = [(r["e_end_mwh"] - battery.e_init) * p_bar * battery.eta_dis if end else 0.0
           for r, end in zip(rows, chain_end)]
    return df.with_columns(pl.Series("chain_end", chain_end), pl.Series("terminal_adj_eur", adj),
                           (pl.col("net_eur") + pl.Series(adj)).alias("net_adj_eur"),
                           pl.lit(p_bar).alias("mean_price_eur_mwh"))
