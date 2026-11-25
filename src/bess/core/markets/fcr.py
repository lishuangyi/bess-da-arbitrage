"""Frequency Containment Reserve (FCR) market - interface only, not used by any pipeline yet.

TODO: verify every rule against regelleistung.net before implementing:
- auction on D-1 in the morning (before DA gate closure), so FCR must be decided
  before the DA bid and its SoC/power occupancy enters the DA problem;
- products are 4-hour blocks (6 per day), symmetric (same MW up and down);
- variables: reserved capacity r_b >= 0 per block b;
- occupancy: p_dis + r <= P_max, p_ch + r <= P_max, and SoC margins so the
  reserve can be delivered for the required duration in both directions;
- revenue: capacity price * r_b * block length; energy activation neglected.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from bess.core.contracts import DayContext, Schedule
from bess.core.markets.base import Market


class FCRMarket(Market):  # pragma: no cover - not implemented in v1
    name = "fcr"
    block_hours = 4

    def gate_closure_utc(self, delivery_day) -> dt.datetime:
        raise NotImplementedError("FCR gate closure: verify current auction time on regelleistung.net")

    def add_variables(self, lp, ctx: DayContext) -> None:
        raise NotImplementedError("TODO v2: add r[block] capacity variables")

    def add_occupancy_constraints(self, lp, ctx: DayContext) -> None:
        raise NotImplementedError("TODO v2: power reservation and SoC margins")

    def add_revenue(self, lp, ctx: DayContext) -> None:
        raise NotImplementedError("TODO v2: capacity price * r * block length")

    def settle(self, schedule: Schedule, realised_prices: np.ndarray) -> np.ndarray:
        raise NotImplementedError
