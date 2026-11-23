"""Abstract market interface.

A market contributes to the day's scheduling problem through four hooks:
- `gate_closure_utc`: when decisions for a delivery day must be fixed;
- `add_variables`: its own decision variables (e.g. FCR capacity per 4-h block);
- `add_occupancy_constraints`: how it occupies battery power and SoC headroom
  (e.g. FCR power reservation and SoC margins);
- `add_revenue`: its objective terms;
and settles a fixed schedule ex post with `settle` (realised prices only).

Both the MILP (`bess.optim.milp`) and the RL environment (`bess.rl.env`) build
their problem through this interface, so the day-ahead market is not hard-coded.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bess.config import Battery


@dataclass
class DayContext:
    """Everything a market needs to formulate one scheduling problem."""

    n: int                 # number of steps in the horizon
    dt_h: float            # step length in hours
    prices: np.ndarray     # price signal used for *decisions* (forecast, or actual for the oracle)
    battery: Battery


@dataclass
class Schedule:
    """Physical battery schedule. Powers in MW (>= 0), energy in MWh (n+1 points)."""

    p_ch: np.ndarray
    p_dis: np.ndarray
    e: np.ndarray
    dt_h: float

    @property
    def n(self) -> int:
        return len(self.p_ch)

    @property
    def p_net(self) -> np.ndarray:
        """Net injection to the grid (discharge positive)."""
        return self.p_dis - self.p_ch


class Market(ABC):
    name: str = "market"

    @abstractmethod
    def gate_closure_utc(self, delivery_day) -> dt.datetime:
        """Time at which the decisions for `delivery_day` are locked."""

    def add_variables(self, lp, ctx: DayContext) -> None:
        """Add market-specific decision variables (none by default)."""

    def add_occupancy_constraints(self, lp, ctx: DayContext) -> None:
        """Constrain battery power/SoC headroom used by this market (none by default)."""

    @abstractmethod
    def add_revenue(self, lp, ctx: DayContext) -> None:
        """Add this market's revenue terms to the (maximisation) objective."""

    @abstractmethod
    def settle(self, schedule: Schedule, realised_prices: np.ndarray) -> np.ndarray:
        """Per-step revenue (EUR) of a fixed schedule at realised prices."""

    def step_limits(self, ctx: DayContext, t: int) -> tuple[float, float]:
        """Power headroom (max charge MW, max discharge MW) left for step t after
        this market's commitments. Used by the RL action projection."""
        b = ctx.battery
        return b.p_max_mw, b.p_max_mw
