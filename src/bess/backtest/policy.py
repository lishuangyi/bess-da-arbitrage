"""Policy interface used by the backtester."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bess.config import Battery
from bess.data.smard import MarketData
from bess.markets.base import Market, Schedule


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
    def plan(self, day, info: MarketData, resolution: str, e_init: float, battery: Battery,
             market: Market, oracle_prices: np.ndarray | None = None) -> Plan:
        """Full-day schedule for `day`, decided with information `info` only.

        `oracle_prices` (realised prices of `day`) is passed only if `uses_oracle`.
        """
