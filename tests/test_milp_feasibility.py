"""MILP feasibility, correctness on a hand-solvable case, and optimality sanity checks."""
import numpy as np
import pytest

from bess.markets.base import Schedule
from bess.optim.milp import count_violations, energy_trajectory, schedule_objective, solve_schedule


@pytest.mark.parametrize("dt_h,n", [(1.0, 24), (1.0, 23), (1.0, 25), (0.25, 96), (0.25, 92), (0.25, 100)])
def test_feasible_random_prices(battery, dt_h, n):
    rng = np.random.default_rng(n)
    for _ in range(5):
        prices = rng.normal(60, 80, n)  # many negative prices
        r = solve_schedule(prices, dt_h, battery)
        s = r.schedule
        assert count_violations(s, battery, battery.e_init, battery.e_final) == 0
        assert abs(s.e[-1] - battery.e_final) < 1e-6
        assert not np.any((s.p_ch > 1e-9) & (s.p_dis > 1e-9))
        assert abs(schedule_objective(s, prices, battery) - r.objective) < 1e-6
        assert r.objective >= -1e-6  # idle is feasible


def test_hand_solved_two_steps(battery):
    # charge at 0 EUR, discharge at 100 EUR; the charge is limited by soc_max:
    # e: 1.0 -> 1.8 MWh needs p_ch = 0.8 / eta_ch; discharged energy = 0.8 * eta_dis
    r = solve_schedule([0.0, 100.0], 1.0, battery)
    e_dis = 0.8 * battery.eta_dis
    assert abs(r.objective - (100.0 - battery.c_deg_eur_per_mwh) * e_dis) < 1e-6
    assert abs(r.schedule.p_ch[0] - 0.8 / battery.eta_ch) < 1e-6


def test_all_negative_prices_no_simultaneous(battery):
    # with negative prices an LP relaxation would charge and discharge at once to burn energy
    prices = np.full(24, -50.0)
    r = solve_schedule(prices, 1.0, battery)
    assert not np.any((r.schedule.p_ch > 1e-9) & (r.schedule.p_dis > 1e-9))
    assert count_violations(r.schedule, battery, battery.e_init, battery.e_final) == 0


def test_hold_constant_power_within_hour(battery):
    rng = np.random.default_rng(3)
    prices = rng.normal(80, 60, 96)
    free = solve_schedule(prices, 0.25, battery)
    held = solve_schedule(prices, 0.25, battery, hold_steps=4)
    for p in (held.schedule.p_ch, held.schedule.p_dis):
        assert np.allclose(p.reshape(24, 4), p.reshape(24, 4)[:, :1], atol=1e-7)
    assert held.objective <= free.objective + 1e-6


def test_perfect_foresight_dominates_any_feasible_schedule(battery):
    rng = np.random.default_rng(7)
    actual = rng.normal(80, 60, 96)
    forecast = actual + rng.normal(0, 40, 96)
    pf = solve_schedule(actual, 0.25, battery)
    fc = solve_schedule(forecast, 0.25, battery)
    assert pf.objective >= schedule_objective(fc.schedule, actual, battery) - 1e-6


def test_violation_checker_detects_errors(battery):
    n = 4
    s = Schedule(np.array([1.0, 1.0, 0, 0]), np.array([0, 0.5, 0, 0.0]), np.zeros(n + 1), 1.0)
    assert count_violations(s, battery, battery.e_init, battery.e_final) > 0
    p = np.full(4, 1.0)
    s = Schedule(p, np.zeros(4), energy_trajectory(p, np.zeros(4), 1.0, battery, battery.e_init), 1.0)
    assert count_violations(s, battery, battery.e_init, None) > 0  # exceeds soc_max


@pytest.mark.parametrize("cycles", [0.5, 1.0, 1.5])
def test_cycle_limit(battery, cycles):
    import dataclasses
    rng = np.random.default_rng(11)
    prices = 80 + 120 * np.sin(np.linspace(0, 8 * np.pi, 96)) + rng.normal(0, 30, 96)  # many cycles pay
    free = solve_schedule(prices, 0.25, battery)
    capped_b = dataclasses.replace(battery, max_cycles_per_day=cycles)
    capped = solve_schedule(prices, 0.25, capped_b)
    e_dis = float(np.sum(capped.schedule.p_dis)) * 0.25
    assert e_dis <= cycles * battery.e_max_mwh + 1e-6
    assert count_violations(capped.schedule, capped_b, battery.e_init, battery.e_final) == 0
    assert capped.objective <= free.objective + 1e-6
    # the unconstrained optimum cycles more than the cap here, so the cap must bind
    assert float(np.sum(free.schedule.p_dis)) * 0.25 > cycles * battery.e_max_mwh


def test_cycle_limit_per_day_in_multi_day_horizon(battery):
    import dataclasses
    rng = np.random.default_rng(5)
    prices = 80 + 120 * np.sin(np.linspace(0, 16 * np.pi, 48)) + rng.normal(0, 30, 48)
    b = dataclasses.replace(battery, max_cycles_per_day=1.0)
    r = solve_schedule(prices, 1.0, b, day_steps=[23, 25])   # DST-length days
    e = r.schedule.p_dis  # dt = 1 h
    assert e[:23].sum() <= b.e_max_mwh + 1e-6 and e[23:].sum() <= b.e_max_mwh + 1e-6
    with pytest.raises(ValueError):
        solve_schedule(prices, 1.0, b, day_steps=[24, 23])


def test_cycle_limit_on_stored_charge():
    """ISEA convention: the cap counts energy added to the store (eta_ch * p_ch * dt)."""
    import dataclasses

    from bess.config import Battery
    from bess.optim.milp import cycle_throughput_mwh

    b = Battery(p_max_mw=1.0, e_max_mwh=2.0, eta_ch=0.95, eta_dis=0.95, soc_min=0.0, soc_max=1.0,
                c_deg_eur_per_mwh=0.0, max_cycles_per_day=1.0, cycle_basis="stored_charge")
    prices = np.tile([10.0, 10.0, 200.0, 200.0], 6)
    r = solve_schedule(prices, 1.0, b)
    stored = 0.95 * float(r.schedule.p_ch.sum())
    assert stored <= 2.0 + 1e-6 and abs(stored - cycle_throughput_mwh(r.schedule, b)) < 1e-9
    assert count_violations(r.schedule, b, b.e_init, b.e_final) == 0
    # on a discharge basis the same cap allows more charging (discharge 2 MWh needs 2/0.95^2 stored)
    rd = solve_schedule(prices, 1.0, dataclasses.replace(b, cycle_basis="discharge"))
    assert 0.95 * float(rd.schedule.p_ch.sum()) > stored + 1e-6
