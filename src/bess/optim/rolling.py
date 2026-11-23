"""Rolling-horizon MILP (MPC form).

At gate closure of D (D-1 12:00) solve a 48 h problem over D and D+1, using the
forecasts for D (horizon 1) and D+1 (horizon 2) that are available at that
time, with the terminal SoC constraint at the end of D+1. Only day D is
executed (and locked in the DA auction); the end-of-day SoC of D is free and is
carried over as the start SoC of D+1.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from bess.backtest.policy import Plan, Policy
from bess.data import calendar as cal
from bess.markets.base import Schedule
from bess.optim.milp import solve_schedule


class RollingMILP(Policy):
    carries_soc = True

    def __init__(self, forecaster, horizon_days: int = 2, hold_steps: int = 1, name: str | None = None):
        if horizon_days < 1:
            raise ValueError("horizon_days >= 1")
        self.forecaster = forecaster
        self.horizon_days = horizon_days
        self.hold_steps = hold_steps
        self.name = name or f"rolling{24 * horizon_days}h_milp_{forecaster.name}"

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None) -> Plan:
        day = cal.to_date(day)
        dt_h = cal.RESOLUTION_HOURS[resolution]
        parts = [self.forecaster.predict(day + dt.timedelta(days=i), info, resolution)
                 for i in range(self.horizon_days)]
        n1 = len(parts[0])
        r = solve_schedule(np.concatenate(parts), dt_h, battery, [market], e_init=e_init,
                           e_final=battery.e_final, hold_steps=self.hold_steps,
                           day_steps=[len(x) for x in parts])
        s = r.schedule
        first = Schedule(s.p_ch[:n1].copy(), s.p_dis[:n1].copy(), s.e[: n1 + 1].copy(), dt_h)
        return Plan(first, r.solve_time_s, None)
