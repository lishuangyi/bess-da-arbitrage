"""Shared contracts between the modules: the day problem (DayContext), a physical schedule (Schedule),
and the policy interface every scheduler implements for the backtester (Plan, Policy)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bess.core.config import Battery

if TYPE_CHECKING:  # type-only: keeps core free of input/ and of the market rules
    from bess.core.markets.base import Market
    from bess.input.prices import MarketData


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


@dataclass
class Plan:
    schedule: Schedule
    compute_time_s: float          # solve / inference time
    e_final_target: float | None   # terminal energy the plan is required to reach (None = free)
    diagnostics: dict | None = None  # extra per-day values written to the backtest row (e.g. action_clip_share)


class Policy(ABC):
    name = "policy"
    uses_oracle = False   # True only for the perfect-foresight upper bound
    carries_soc = False   # True if the end-of-day SoC is free and carried to the next day

    @abstractmethod
    def plan(self, day, info: "MarketData", resolution: str, e_init: float, battery: Battery,
             market: "Market", oracle_prices: np.ndarray | None = None) -> Plan:
        """Full-day schedule for `day`, decided with information `info` only.

        `oracle_prices` (realised prices of `day`) is passed only if `uses_oracle`.
        """
