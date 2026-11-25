"""Daily MILP policies for the backtester: perfect foresight (oracle reference) and the forecast-based MILP."""
from __future__ import annotations

from bess.core import calendar as cal
from bess.core.contracts import Plan, Policy
from bess.milp.model import solve_schedule


class PerfectForesight(Policy):
    """Upper bound: daily MILP on realised prices (oracle)."""

    uses_oracle = True

    def __init__(self, hold_steps: int = 1, name: str = "perfect_foresight"):
        self.hold_steps = hold_steps
        self.name = name

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None) -> Plan:
        assert oracle_prices is not None
        r = solve_schedule(oracle_prices, cal.RESOLUTION_HOURS[resolution], battery, [market],
                           e_init=e_init, hold_steps=self.hold_steps)
        return Plan(r.schedule, r.solve_time_s, battery.e_final)


class ForecastMILP(Policy):
    """Daily MILP on a price forecast made at gate closure; settled at realised prices."""

    def __init__(self, forecaster, hold_steps: int = 1, name: str | None = None):
        self.forecaster = forecaster
        self.hold_steps = hold_steps
        self.name = name or f"forecast_milp_{forecaster.name}"

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None) -> Plan:
        f = self.forecaster.predict(day, info, resolution)
        r = solve_schedule(f, cal.RESOLUTION_HOURS[resolution], battery, [market],
                           e_init=e_init, hold_steps=self.hold_steps)
        return Plan(r.schedule, r.solve_time_s, battery.e_final)
