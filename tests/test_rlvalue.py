"""v2 (learned end-of-day SoC value): reachability, reward cache, features, chained backtest."""
import datetime as dt

import numpy as np
import pytest

from bess.data import calendar as cal
from bess.forecast.naive import SeasonalNaive
from bess.markets.day_ahead import DayAheadMarket
from bess.optim.milp import ForecastMILP, TerminalValue, count_violations, schedule_objective, solve_schedule
from bess.rlvalue import cache as cachemod
from bess.rlvalue.cache import gate_closure_inputs, level_energy, mid_index, reward_tensor, soc_levels
from bess.rlvalue.features import FEATURE_NAMES

from test_no_lookahead import perturbed_from

QH = dt.date(2025, 10, 1)


def _isea_battery():
    from bess.config import Battery
    return Battery.from_yaml("battery_isea.yaml")


@pytest.mark.parametrize("profile", ["project", "isea"])
@pytest.mark.parametrize("day,dt_h", [("2025-03-30", 1.0), ("2025-07-01", 1.0), ("2025-10-26", 1.0),
                                      ("2026-03-29", 0.25), ("2026-01-15", 0.25), ("2025-10-26", 0.25)])
def test_every_level_pair_reachable_within_one_day(battery, day, dt_h, profile):
    """Project battery (SoC 10-90 %, 9 levels) and ISEA battery (0-100 %, 11 levels, 2 cycles/day on
    stored charge): any end level can be reached from any start level within a day, incl. 23-hour days."""
    battery = battery if profile == "project" else _isea_battery()
    rng = np.random.default_rng(0)
    prices = rng.normal(80, 60, cal.expected_steps(day, dt_h))
    lv = level_energy(battery)
    for e0 in lv:
        for e1 in lv:
            r = solve_schedule(prices, dt_h, battery, e_init=float(e0), e_final=float(e1))
            assert count_violations(r.schedule, battery, float(e0), float(e1)) == 0
            assert abs(r.schedule.e[-1] - e1) < 1e-6
    lv_frac = soc_levels(battery)
    assert len(lv_frac) == (9 if profile == "project" else 11) and abs(lv_frac[mid_index(lv_frac)] - 0.5) < 1e-12


def _inputs(mdata, days):
    return gate_closure_inputs(mdata, SeasonalNaive(), days, QH, "12:00", "13:00")


def test_cache_matches_direct_solve_and_fixed_50_percent_equals_daily_milp(mdata, battery):
    from bess.backtest.engine import run_backtest

    days = [dt.date(2025, 7, 1), dt.date(2026, 1, 12)]
    inputs = _inputs(mdata, days)
    cachemod._init_worker(battery)
    lv = level_energy(battery)
    MID = mid_index(soc_levels(battery))
    for d in days:
        res, f, _ = inputs[d]
        actual = mdata.series(res).day_prices(d)
        rows = {(r["s"], r["e"]): r for r in cachemod.solve_day((d, res, f, actual))}
        assert len(rows) == 81
        for s, e in [(0, 8), (8, 0), (3, 6), (MID, MID)]:
            r = solve_schedule(f, cal.RESOLUTION_HOURS[res], battery, e_init=float(lv[s]), e_final=float(lv[e]))
            assert abs(rows[(s, e)]["net_eur"] - schedule_objective(r.schedule, actual, battery)) < 1e-6
            assert abs(rows[(s, e)]["forecast_obj_eur"] - r.objective) < 1e-6
        # start = end = 50 % is the v1 daily MILP (E2) with the same forecaster
        e2 = run_backtest(ForecastMILP(SeasonalNaive()), mdata, [d], res, battery, DayAheadMarket())
        assert abs(rows[(MID, MID)]["net_eur"] - e2["net_eur"][0]) < 1e-6


def test_reward_tensor_layout(mdata, battery):
    import polars as pl

    d = dt.date(2025, 7, 1)
    res, f, _ = _inputs(mdata, [d])[d]
    cachemod._init_worker(battery)
    c = pl.DataFrame(cachemod.solve_day((d, res, f, mdata.series(res).day_prices(d))))
    R = reward_tensor(c.sample(fraction=1.0, shuffle=True, seed=1), [d])
    assert R.shape == (1, 9, 9)
    assert R[0, 2, 7] == c.filter((pl.col("s") == 2) & (pl.col("e") == 7))["net_eur"][0]


@pytest.mark.parametrize("day", [dt.date(2025, 9, 30), dt.date(2025, 10, 26), dt.date(2026, 1, 10)])
def test_state_features_use_only_gate_closure_information(mdata, day):
    """phi_D must not change when the realised prices of D (and later) change."""
    a = _inputs(mdata, [day])[day]
    b = _inputs(perturbed_from(mdata, day), [day])[day]
    assert a[0] == b[0] and len(a[2]) == len(FEATURE_NAMES)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert a[2][FEATURE_NAMES.index("qh_flag")] == float(day >= QH)


def test_forward_fold_model_is_chosen_by_decision_day(mdata):
    """Synthetic data start in Feb 2025: only the 2026 fold has history. The forecast of D+1 = 1 Jan
    made at gate closure of D = 31 Dec would need the 2025 model, so that day has no inputs; from
    D = 1 Jan 2026 on both horizons use the 2026 model, which never saw 2026 prices."""
    from bess.forecast import xgb as xgbmod
    from bess.rlvalue.forecasts import ForwardFoldXGB

    old = dict(xgbmod.XGB_PARAMS)
    xgbmod.XGB_PARAMS.update(n_estimators=20, early_stopping_rounds=5)
    try:
        fc = ForwardFoldXGB().fit(mdata, [2025, 2026])
        pert = ForwardFoldXGB().fit(perturbed_from(mdata, dt.date(2026, 1, 1)), [2025, 2026])
    finally:
        xgbmod.XGB_PARAMS.clear()
        xgbmod.XGB_PARAMS.update(old)
    assert fc.years() == [2026]
    days = [dt.date(2025, 12, 31), dt.date(2026, 1, 1), dt.date(2026, 1, 5)]
    inp = gate_closure_inputs(mdata, fc, days, QH, "12:00", "13:00")
    assert sorted(inp) == days[1:]
    d = dt.date(2026, 1, 1)                      # same model whatever the 2026 prices are
    X = np.zeros((3, len(xgbmod.FEATURES)))
    assert np.array_equal(fc.models[(2026, 1)].predict(X), pert.models[(2026, 1)].predict(X))
    assert np.array_equal(inp[d][1], gate_closure_inputs(mdata, pert, [d], QH, "12:00", "13:00")[d][1])


def test_chain_carries_soc_restarts_at_gaps_and_values_the_end(mdata, battery):
    from bess.backtest.chain import run_chain
    from bess.rlvalue.policy import LinearResidualMILP

    days = [dt.date(2026, 1, d) for d in (10, 11, 12, 14, 15)]           # gap: 13 Jan
    days_res = [(d, "quarterhour") for d in days]
    pol = LinearResidualMILP(SeasonalNaive(), eur_per_mwh=60.0)
    df = run_chain(pol, mdata, days_res, battery, DayAheadMarket(), mean_price=100.0)
    e0, e1 = df["e_start_mwh"].to_list(), df["e_end_mwh"].to_list()
    assert df["chain_start"].to_list() == [True, False, False, True, False]
    assert df["chain_end"].to_list() == [False, False, True, False, True]
    assert e0[0] == e0[3] == battery.e_init
    assert e0[1] == e1[0] and e0[2] == e1[1] and e0[4] == e1[3]
    adj = df["terminal_adj_eur"].to_list()
    assert adj[0] == adj[1] == adj[3] == 0.0
    assert abs(adj[2] - (e1[2] - battery.e_init) * 100.0 * battery.eta_dis) < 1e-9
    assert (df["violations"] == 0).all()
    assert np.allclose(df["net_adj_eur"].to_numpy(), (df["net_eur"] + df["terminal_adj_eur"]).to_numpy())


def test_level_terminal_value_picks_best_level(battery):
    """MILP with a value per end level = best of the fixed-end MILPs plus that level's value."""
    rng = np.random.default_rng(3)
    prices = rng.normal(80, 70, 24)
    lv = level_energy(battery)
    w = rng.normal(0, 40, 9)
    tv = TerminalValue(levels_mwh=tuple(lv), values_eur=tuple(w))
    r = solve_schedule(prices, 1.0, battery, e_init=float(lv[2]), e_final=None, terminal=tv)
    fixed = [solve_schedule(prices, 1.0, battery, e_init=float(lv[2]), e_final=float(e)).objective for e in lv]
    best = int(np.argmax(np.asarray(fixed) + w))
    assert abs(r.objective - (fixed[best] + w[best])) < 1e-5
    assert abs(r.schedule.e[-1] - lv[best]) < 1e-6 and abs(r.terminal_eur - w[best]) < 1e-12


def test_fqi_matches_tabular_value_iteration():
    """Constant features and the same reward every day: FQI must reproduce tabular value iteration
    Q = R + gamma * max_a Q(e, a) on the 9 x 9 levels, and gamma = 0 must return R."""
    from bess.rlvalue.fqi import FQIData, continuation_values, fit_fqi, fit_projection, q_inputs

    rng = np.random.default_rng(0)
    n, gamma, K = 40, 0.8, 25
    levels = np.round(np.linspace(0.1, 0.9, 9), 10)
    R1 = rng.normal(0, 1, (9, 9))
    d = FQIData(list(range(n)), np.zeros((n, 2)), np.repeat(R1[None], n, axis=0),
                np.array(list(range(1, n)) + [-1]), levels)
    # 39 of 40 rows bootstrap, the last day is terminal: identical features average the two,
    # i.e. tabular value iteration with an effective discount gamma * 39 / 40
    g_eff = gamma * (n - 1) / n
    Q = np.zeros((9, 9))
    for _ in range(K):
        Q = R1 + g_eff * Q.max(axis=1)[None, :]
    params = {"n_estimators": 5, "min_samples_leaf": 1}
    res = fit_fqi(d, gamma, K, "extra_trees", params, seed=0, scale=1.0)
    q_hat = res.models[K].predict(q_inputs(d.X[:1], levels)).reshape(9, 9)
    assert np.abs(q_hat - Q).max() < 1e-6 and res.deltas[-1] < res.deltas[0]
    r0 = fit_fqi(d, 0.0, K, "extra_trees", params, seed=0, scale=1.0)
    assert list(r0.models) == [1]
    assert np.allclose(r0.models[1].predict(q_inputs(d.X[:1], levels)).reshape(9, 9), R1, atol=1e-9)
    # projection: W(e) = V(e) - V(50 %) of the next day
    w = fit_projection(d, res.models[K], "extra_trees", params, seed=0)
    V = continuation_values(res.models[K], d.X[:1], levels)[0]
    assert np.allclose(w.predict(np.column_stack([np.zeros((9, 2)), np.linspace(0.1, 0.9, 9)])), V - V[4], atol=1e-6)
