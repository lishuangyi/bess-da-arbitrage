"""Benchmark protocol sanity checks on synthetic daily results."""
import datetime as dt

import numpy as np
import polars as pl

from bess.backtest.benchmark import block_bootstrap_means, run_benchmark


def _daily(net, viol=0):
    n = len(net)
    return pl.DataFrame({"day": [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)],
                         "net_eur": net, "efc": np.ones(n),
                         "violations": [viol] + [0] * (n - 1), "compute_time_s": np.full(n, 0.01)})


def test_block_bootstrap_centred_and_deterministic():
    x = np.random.default_rng(0).normal(5, 2, 200)
    a, b = block_bootstrap_means(x, seed=1), block_bootstrap_means(x, seed=1)
    assert np.array_equal(a, b)
    assert abs(a.mean() - x.mean()) < 0.1


def test_ranking_tiers_and_gate():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 30, 120)
    ref = _daily(base + 60)
    daily = {
        "perfect_foresight": ref,
        "good": _daily(base + 20),
        "good_twin": _daily(base + 20 + rng.normal(0, 1, 120)),   # statistically tied with 'good'
        "bad": _daily(base - 30),
        "infeasible": _daily(base + 50, viol=3),
    }
    bm = {r["method"]: r for r in run_benchmark(daily, ref, n_boot=2000).to_dicts()}
    assert bm["perfect_foresight"]["tier"] == "reference"
    assert bm["infeasible"]["tier"] == "disqualified"
    assert {bm["good"]["tier"], bm["good_twin"]["tier"]} == {"1 (leader)", "1 (tied)"}
    assert bm["bad"]["tier"] == "2+" and bm["bad"]["rank"] == 3
    assert bm["bad"]["diff_ci95_hi"] < 0


def test_seeded_result_files_are_kept_apart_and_averaged(tmp_path):
    import datetime as dt

    import polars as pl

    from bess.backtest.io import load_daily_dir, load_daily_seeds

    days = [dt.date(2026, 1, 1), dt.date(2026, 1, 2)]
    base = {"day": days, "n_steps": [96, 96], "efc": [1.0, 1.0], "compute_time_s": [0.1, 0.1]}
    pl.DataFrame({**base, "net_eur": [10.0, 20.0], "violations": [0, 0]}).write_csv(tmp_path / "E5__ppo_xgb__test__seed0.csv")
    pl.DataFrame({**base, "net_eur": [30.0, 40.0], "violations": [0, 1]}).write_csv(tmp_path / "E5__ppo_xgb__test__seed1.csv")
    pl.DataFrame({**base, "net_eur": [5.0, 5.0], "violations": [0, 0]}).write_csv(tmp_path / "E2__forecast_milp_xgb__test.csv")
    seeds = load_daily_seeds(tmp_path, "test")
    assert sorted(seeds[("E5", "ppo_xgb", "test")]) == [0, 1]
    assert list(seeds[("E2", "forecast_milp_xgb", "test")]) == [None]
    mean = load_daily_dir(tmp_path, "test")[("E5", "ppo_xgb", "test")]
    assert mean["net_eur"].to_list() == [20.0, 30.0] and mean["violations"].to_list() == [0, 1]


def test_block_bootstrap_ratio_pairs_days():
    import numpy as np

    from bess.backtest.benchmark import block_bootstrap_ratio

    rng = np.random.default_rng(1)
    den = rng.uniform(50, 150, 200)
    r = block_bootstrap_ratio(0.8 * den, den)          # same days resampled on both sides
    assert np.allclose(r, 0.8)
    r2 = block_bootstrap_ratio(den + rng.normal(0, 30, 200), den)
    assert 0.9 < np.median(r2) < 1.1 and r2.std() > 0
