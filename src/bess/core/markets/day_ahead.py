"""Day-ahead (DA) energy market, DE-LU bidding zone (price taker)."""
from __future__ import annotations

import datetime as dt

import numpy as np

from bess.core import calendar as cal
from bess.core.contracts import DayContext, Schedule
from bess.core.markets.base import Market


class DayAheadMarket(Market):
    """Single-price DA auction. Gate closure D-1 12:00 Europe/Berlin.

    The DA position equals the battery's net injection p_dis - p_ch in every
    delivery period, so the market adds no variables of its own and no extra
    occupancy; revenue is price * (p_dis - p_ch) * dt.
    """

    name = "day_ahead"

    def __init__(self, gate_closure_local: str = "12:00"):
        self.gate_closure_local = gate_closure_local

    def gate_closure_utc(self, delivery_day) -> dt.datetime:
        return cal.gate_closure_utc(delivery_day, self.gate_closure_local)

    def add_revenue(self, lp, ctx: DayContext) -> None:
        coef = np.asarray(ctx.prices, dtype=float) * ctx.dt_h
        lp.add_objective("p_dis", coef)
        lp.add_objective("p_ch", -coef)

    def settle(self, schedule: Schedule, realised_prices: np.ndarray) -> np.ndarray:
        return np.asarray(realised_prices, dtype=float) * (schedule.p_dis - schedule.p_ch) * schedule.dt_h

    def step_revenue(self, price: float, p_ch: float, p_dis: float, dt_h: float) -> float:
        return float(price) * (p_dis - p_ch) * dt_h
