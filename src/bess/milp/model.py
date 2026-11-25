"""Battery scheduling MILP (scipy.optimize.milp / HiGHS).

max  sum_m revenue_m  -  c_deg * sum_t p_dis[t] * dt
s.t. e[t+1] = e[t] + eta_ch * p_ch[t] * dt - p_dis[t] * dt / eta_dis
     e[0] = e_init,  e[n] = e_final (optional)
     e_min <= e <= e_max,  0 <= p_ch, p_dis <= P
     p_ch <= P * u,  p_dis <= P * (1 - u),  u binary   (no simultaneous charge/discharge;
                                                       needed with negative prices)
     + market variables / occupancy constraints (see bess.core.markets.base.Market)
     + optional hold constraint: power constant within blocks of `hold_steps`.
     + optional terminal value of the end-of-horizon energy e[n] (see TerminalValue):
       linear (EUR per MWh stored, e[n] free) or on discrete levels (one binary per level,
       e[n] = sum_k z_k * level_k, objective + sum_k z_k * value_k).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from bess.core.config import Battery
from bess.core.contracts import DayContext, Schedule
from bess.core.markets.base import Market
from bess.core.markets.day_ahead import DayAheadMarket


class LinearProgram:
    """Minimal named-block MILP builder (maximisation)."""

    def __init__(self):
        self.blocks: dict[str, slice] = {}
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integrality: list[int] = []
        self.c: list[float] = []
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self.con_lb: list[float] = []
        self.con_ub: list[float] = []

    @property
    def n_vars(self) -> int:
        return len(self.lb)

    def add_var(self, name: str, size: int, lb, ub, integer: bool = False) -> slice:
        assert name not in self.blocks, name
        s = slice(self.n_vars, self.n_vars + size)
        self.blocks[name] = s
        self.lb.extend(np.broadcast_to(np.asarray(lb, float), size).tolist())
        self.ub.extend(np.broadcast_to(np.asarray(ub, float), size).tolist())
        self.integrality.extend([1 if integer else 0] * size)
        self.c.extend([0.0] * size)
        return s

    def idx(self, name: str, i: int | None = None):
        s = self.blocks[name]
        return s if i is None else s.start + i

    def add_objective(self, name: str, coef) -> None:
        s = self.blocks[name]
        coef = np.broadcast_to(np.asarray(coef, float), s.stop - s.start)
        for k, v in enumerate(coef):
            self.c[s.start + k] += float(v)

    def add_row(self, terms: list[tuple[int, float]], lb: float, ub: float) -> None:
        r = len(self.con_lb)
        for j, v in terms:
            self._rows.append(r)
            self._cols.append(j)
            self._vals.append(float(v))
        self.con_lb.append(lb)
        self.con_ub.append(ub)

    def solve(self, time_limit: float | None = None, mip_rel_gap: float = 1e-9):
        A = coo_matrix((self._vals, (self._rows, self._cols)), shape=(len(self.con_lb), self.n_vars)).tocsr()
        opts = {"mip_rel_gap": mip_rel_gap, "disp": False}
        if time_limit is not None:
            opts["time_limit"] = time_limit
        t0 = time.perf_counter()
        res = milp(
            c=-np.asarray(self.c),  # scipy minimises
            constraints=LinearConstraint(A, self.con_lb, self.con_ub),
            integrality=np.asarray(self.integrality),
            bounds=Bounds(self.lb, self.ub),
            options=opts,
        )
        return res, time.perf_counter() - t0


def add_battery(lp: LinearProgram, ctx: DayContext, e_init: float, e_final: float | None,
                day_steps: list[int] | None = None) -> None:
    """Battery variables and constraints. `day_steps`: steps per delivery day in the horizon
    (for the daily cycle limit); None = the horizon is one delivery day."""
    b, n, dt = ctx.battery, ctx.n, ctx.dt_h
    P = b.p_max_mw
    lp.add_var("p_ch", n, 0.0, P)
    lp.add_var("p_dis", n, 0.0, P)
    lp.add_var("e", n + 1, b.e_min, b.e_upper)
    lp.add_var("u", n, 0.0, 1.0, integer=True)
    for t in range(n):
        lp.add_row([(lp.idx("e", t + 1), 1.0), (lp.idx("e", t), -1.0),
                    (lp.idx("p_ch", t), -b.eta_ch * dt), (lp.idx("p_dis", t), dt / b.eta_dis)], 0.0, 0.0)
        lp.add_row([(lp.idx("p_ch", t), 1.0), (lp.idx("u", t), -P)], -np.inf, 0.0)
        lp.add_row([(lp.idx("p_dis", t), 1.0), (lp.idx("u", t), P)], -np.inf, P)
    lp.add_row([(lp.idx("e", 0), 1.0)], e_init, e_init)
    if e_final is not None:
        lp.add_row([(lp.idx("e", n), 1.0)], e_final, e_final)
    if b.max_cycles_per_day is not None:
        day_steps = day_steps or [n]
        if sum(day_steps) != n:
            raise ValueError(f"day_steps sum to {sum(day_steps)}, horizon has {n} steps")
        name, coef = cycle_terms(b, dt)
        start = 0
        for k in day_steps:
            lp.add_row([(lp.idx(name, t), coef) for t in range(start, start + k)], -np.inf,
                       discharge_cap_mwh(b))
            start += k
    lp.add_objective("p_dis", -b.c_deg_eur_per_mwh * dt)


def discharge_cap_mwh(battery: Battery) -> float:
    """Cycle-throughput cap per delivery day (23, 24 or 25 h) from battery.max_cycles_per_day (MWh)."""
    return battery.max_cycles_per_day * battery.e_max_mwh


def cycle_terms(battery: Battery, dt_h: float) -> tuple[str, float]:
    """Variable and coefficient whose sum over a day is the throughput counted by the cycle limit."""
    if battery.cycle_basis == "discharge":
        return "p_dis", dt_h
    if battery.cycle_basis == "stored_charge":
        return "p_ch", battery.eta_ch * dt_h
    raise ValueError(f"unknown cycle_basis {battery.cycle_basis!r}")


def cycle_throughput_mwh(schedule: Schedule, battery: Battery) -> float:
    if battery.cycle_basis == "stored_charge":
        return battery.eta_ch * float(np.sum(schedule.p_ch)) * schedule.dt_h
    return float(np.sum(schedule.p_dis)) * schedule.dt_h


def add_hold(lp: LinearProgram, n: int, hold_steps: int) -> None:
    """Force constant power within consecutive blocks of `hold_steps` steps."""
    for t in range(n):
        if t % hold_steps:
            for name in ("p_ch", "p_dis"):
                lp.add_row([(lp.idx(name, t), 1.0), (lp.idx(name, t - 1), -1.0)], 0.0, 0.0)


@dataclass(frozen=True)
class TerminalValue:
    """Value of the energy left at the end of the horizon, added to the objective.

    Linear: `eur_per_mwh` * e[n] with e[n] free in [e_min, e_upper].
    Levels: e[n] restricted to `levels_mwh`, objective + values_eur[k] for the chosen level.
    """
    eur_per_mwh: float | None = None
    levels_mwh: tuple[float, ...] | None = None
    values_eur: tuple[float, ...] | None = None

    def __post_init__(self):
        lin, lev = self.eur_per_mwh is not None, self.levels_mwh is not None
        if lin == lev:
            raise ValueError("give either eur_per_mwh or levels_mwh/values_eur")
        if lev and (self.values_eur is None or len(self.values_eur) != len(self.levels_mwh)):
            raise ValueError("values_eur must have one value per level")


def add_terminal_value(lp: LinearProgram, n: int, tv: TerminalValue) -> None:
    e_end = lp.idx("e", n)
    if tv.eur_per_mwh is not None:
        lp.c[e_end] += float(tv.eur_per_mwh)
        return
    k = len(tv.levels_mwh)
    lp.add_var("z_end", k, 0.0, 1.0, integer=True)
    lp.add_row([(lp.idx("z_end", i), 1.0) for i in range(k)], 1.0, 1.0)
    lp.add_row([(e_end, 1.0)] + [(lp.idx("z_end", i), -float(tv.levels_mwh[i])) for i in range(k)], 0.0, 0.0)
    lp.add_objective("z_end", np.asarray(tv.values_eur, float))


@dataclass
class MILPResult:
    schedule: Schedule
    objective: float       # value w.r.t. the prices given to the solver
    solve_time_s: float
    status: int
    message: str
    terminal_eur: float = 0.0  # terminal-value part of `objective` (0 without a terminal value)


def solve_schedule(prices, dt_h: float, battery: Battery, markets: list[Market] | None = None,
                   e_init: float | None = None, e_final: float | None = "default",
                   hold_steps: int = 1, time_limit: float | None = 60.0,
                   day_steps: list[int] | None = None, terminal: TerminalValue | None = None) -> MILPResult:
    """Solve the scheduling MILP over len(prices) steps.

    e_final="default" -> battery.e_final; None -> free terminal SoC.
    day_steps: steps per delivery day if the horizon spans several days (daily cycle limit).
    terminal: optional value of the end energy (requires e_final=None). The returned `objective`
    includes it; `MILPResult.terminal_eur` reports the terminal part separately.
    """
    prices = np.asarray(prices, dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("prices contain NaN/inf")
    markets = markets or [DayAheadMarket()]
    n = len(prices)
    ctx = DayContext(n=n, dt_h=dt_h, prices=prices, battery=battery)
    e0 = battery.e_init if e_init is None else e_init
    eT = battery.e_final if e_final == "default" else e_final
    if terminal is not None and eT is not None:
        raise ValueError("a terminal value needs a free end SoC (e_final=None)")
    lp = LinearProgram()
    add_battery(lp, ctx, e0, eT, day_steps)
    if terminal is not None:
        add_terminal_value(lp, n, terminal)
    for m in markets:
        m.add_variables(lp, ctx)
        m.add_occupancy_constraints(lp, ctx)
        m.add_revenue(lp, ctx)
    if hold_steps > 1:
        add_hold(lp, n, hold_steps)
    res, t = lp.solve(time_limit=time_limit)
    if res.x is None:
        raise RuntimeError(f"MILP failed: status={res.status} {res.message}")
    x = res.x
    p_ch = np.clip(x[lp.blocks["p_ch"]], 0.0, None)
    p_dis = np.clip(x[lp.blocks["p_dis"]], 0.0, None)
    # exact complementarity: with u binary one of the two is zero up to solver tolerance
    u = np.round(x[lp.blocks["u"]])
    p_ch = np.where(u > 0.5, p_ch, 0.0)
    p_dis = np.where(u > 0.5, 0.0, p_dis)
    e = energy_trajectory(p_ch, p_dis, dt_h, battery, e0)
    term = 0.0
    if terminal is not None:
        if terminal.eur_per_mwh is not None:
            term = float(terminal.eur_per_mwh) * float(x[lp.idx("e", n)])
        else:
            k = int(np.argmax(x[lp.blocks["z_end"]]))
            term = float(terminal.values_eur[k])
            if abs(e[-1] - terminal.levels_mwh[k]) > 1e-6:
                raise RuntimeError(f"end energy {e[-1]:.6f} differs from chosen level {terminal.levels_mwh[k]}")
    return MILPResult(Schedule(p_ch, p_dis, e, dt_h), -float(res.fun), t, int(res.status), str(res.message), term)


def energy_trajectory(p_ch, p_dis, dt_h: float, battery: Battery, e0: float) -> np.ndarray:
    """Recompute the stored-energy path from the powers (MWh, n+1 points)."""
    delta = battery.eta_ch * np.asarray(p_ch) * dt_h - np.asarray(p_dis) * dt_h / battery.eta_dis
    return np.concatenate([[e0], e0 + np.cumsum(delta)])


def schedule_objective(schedule: Schedule, prices, battery: Battery, markets: list[Market] | None = None) -> float:
    """Objective value of a fixed schedule at `prices` (same expression as the MILP)."""
    markets = markets or [DayAheadMarket()]
    rev = sum(float(np.sum(m.settle(schedule, prices))) for m in markets)
    return rev - battery.c_deg_eur_per_mwh * float(np.sum(schedule.p_dis)) * schedule.dt_h


def count_violations(schedule: Schedule, battery: Battery, e_init: float, e_final: float | None,
                     tol: float = 1e-6) -> int:
    """Number of violated physical constraints, re-derived from the powers."""
    P = battery.p_max_mw
    pc, pd_ = schedule.p_ch, schedule.p_dis
    e = energy_trajectory(pc, pd_, schedule.dt_h, battery, e_init)
    v = 0
    v += int(np.sum((pc < -tol) | (pc > P + tol)))
    v += int(np.sum((pd_ < -tol) | (pd_ > P + tol)))
    v += int(np.sum((pc > tol) & (pd_ > tol)))
    v += int(np.sum((e < battery.e_min - tol) | (e > battery.e_upper + tol)))
    if e_final is not None:
        v += int(abs(e[-1] - e_final) > tol)
    if battery.max_cycles_per_day is not None:
        v += int(cycle_throughput_mwh(schedule, battery) > discharge_cap_mwh(battery) + tol)  # one delivery day
    return v
