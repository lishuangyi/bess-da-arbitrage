"""Daily MILP policies with a value for the end-of-day SoC (v2), for the chained backtest.

- LinearResidualMILP (B1, B2): free end-of-day SoC valued at a constant EUR per MWh stored
  (B1: 0, i.e. no value; B2: constant tuned on the validation period).
- LearnedValueMILP (M1; B3 with a gamma = 0 model): end level valued by the learned W(phi_D, e).
- LearnedEndLevelMILP (M2): the Q function picks the end level, the MILP plans the day to it.
"""
from __future__ import annotations

from bess.core import calendar as cal
from bess.core.contracts import Plan, Policy
from bess.milp.model import TerminalValue, solve_schedule
from bess.rl.value.cache import soc_levels


class LinearResidualMILP(Policy):
    carries_soc = True

    def __init__(self, forecaster, eur_per_mwh: float, name: str | None = None):
        self.forecaster = forecaster
        self.eur_per_mwh = float(eur_per_mwh)
        self.name = name or f"milp_{forecaster.name}_residual{self.eur_per_mwh:g}"

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None) -> Plan:
        f = self.forecaster.predict(day, info, resolution)
        r = solve_schedule(f, cal.RESOLUTION_HOURS[resolution], battery, [market], e_init=e_init, e_final=None,
                           terminal=TerminalValue(eur_per_mwh=self.eur_per_mwh))
        return Plan(r.schedule, r.solve_time_s, None)


class _GateClosureState:
    """phi_D from the policy's forecaster at gate closure of D (same as the reward cache)."""

    def __init__(self, forecaster, qh_start):
        self.forecaster, self.qh_start = forecaster, qh_start

    def __call__(self, day, info, resolution):
        import datetime as dt

        from bess.rl.value.cache import resolution_of
        from bess.rl.value.features import state_features

        day = cal.to_date(day)
        nxt = day + dt.timedelta(days=1)
        res1 = resolution_of(nxt, self.qh_start)
        f0 = self.forecaster.predict(day, info, resolution)
        f1 = self.forecaster.predict(nxt, info, res1)
        phi = state_features(day, f0, cal.RESOLUTION_HOURS[resolution], f1, cal.RESOLUTION_HOURS[res1])
        return f0, phi


class LearnedValueMILP(Policy):
    """M1 (and B3 with a gamma = 0 model): daily MILP whose end level gets the learned value
    W(phi_D, e) in EUR; the end of day is restricted to the 9 SoC levels."""

    carries_soc = True

    def __init__(self, forecaster, w_model, scale: float, qh_start, name: str = "M1_learned_value"):
        self.state = _GateClosureState(forecaster, qh_start)
        self.w_model, self.scale, self.name = w_model, float(scale), name

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None) -> Plan:
        import time

        import numpy as np

        from bess.rl.value.fqi import w_inputs

        t0 = time.perf_counter()
        f, phi = self.state(day, info, resolution)
        levels = soc_levels(battery)
        w = self.w_model.predict(w_inputs(phi[None, :], levels)) * self.scale
        lv = levels * battery.e_max_mwh
        r = solve_schedule(f, cal.RESOLUTION_HOURS[resolution], battery, [market], e_init=e_init, e_final=None,
                           terminal=TerminalValue(levels_mwh=tuple(lv), values_eur=tuple(np.asarray(w, float))))
        k = int(np.argmin(np.abs(lv - r.schedule.e[-1])))
        return Plan(r.schedule, time.perf_counter() - t0, float(lv[k]),
                    diagnostics={"end_level": round(100 * float(levels[k])), "terminal_value_eur": r.terminal_eur})


class LearnedEndLevelMILP(Policy):
    """M2: the Q function picks the end level, argmax_a Q(phi_D, e_s, a); the MILP plans the day to it."""

    carries_soc = True

    def __init__(self, forecaster, q_model, qh_start, name: str = "M2_rl_end_level"):
        self.state = _GateClosureState(forecaster, qh_start)
        self.q_model, self.name = q_model, name

    def plan(self, day, info, resolution, e_init, battery, market, oracle_prices=None) -> Plan:
        import time

        import numpy as np

        from bess.rl.value.fqi import q_inputs

        t0 = time.perf_counter()
        f, phi = self.state(day, info, resolution)
        levels = soc_levels(battery)
        lv = levels * battery.e_max_mwh
        s = int(np.argmin(np.abs(lv - e_init)))
        if abs(lv[s] - e_init) > 1e-6:
            raise ValueError(f"start energy {e_init} is not on the SoC grid")
        n = len(levels)
        q = self.q_model.predict(q_inputs(phi[None, :], levels)).reshape(n, n)[s]
        a = int(np.argmax(q))
        r = solve_schedule(f, cal.RESOLUTION_HOURS[resolution], battery, [market], e_init=e_init,
                           e_final=float(lv[a]))
        return Plan(r.schedule, time.perf_counter() - t0, float(lv[a]),
                    diagnostics={"end_level": round(100 * float(levels[a]))})
