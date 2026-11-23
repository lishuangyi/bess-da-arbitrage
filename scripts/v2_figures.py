"""v2 figures: learned storage value, cumulative differences, FQI convergence.

Usage: python scripts/v2_figures.py [--period test]
Writes results/figures/v2_value_curves.png, v2_cumulative_diff_<period>.png, v2_fqi_convergence.png.
The ablation table itself is results/v2/summary_<period>.csv (scripts/v2_summary.py).
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from bess.backtest.io import load_daily_seeds, mean_over_seeds  # noqa: E402
from bess.config import REPO_ROOT, Battery  # noqa: E402
from bess.rlvalue.cache import soc_levels  # noqa: E402
from bess.rlvalue.features import FEATURE_NAMES  # noqa: E402
from bess.rlvalue.fqi import w_inputs  # noqa: E402
from bess.rlvalue.periods import add_v2_config_arg, v2_battery, v2_config, v2_dir  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"M1_learned_value": "#2a78d6", "B3_one_step_value": "#eb6834", "B4_rolling48h": "#eda100",
       "B0_fixed50": "#1baf7a", "B2_constant_value": "#6250d6", "M2_rl_end_level": "#e87ba4"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def value_curves(models_dir, feats: pl.DataFrame, period: tuple, out, prefix: str, battery) -> None:
    """W(phi_D, e) in EUR (relative to 50 %) for three kinds of days of the period."""
    with open(models_dir / "fqi_M1_seed0.pkl", "rb") as f:
        m = pickle.load(f)
    lv = np.asarray(m.get("levels", soc_levels(battery)))
    f = feats.filter((pl.col("day") >= period[0]) & (pl.col("day") <= period[1]))
    spread = f["d1_peak_valley"].to_numpy()
    days = f["day"].to_list()
    picks = {"Friday (weekend ahead)": next(i for i, d in enumerate(days) if d.weekday() == 4),
             "largest D+1 forecast spread": int(np.argmax(spread)),
             "smallest D+1 forecast spread": int(np.argmin(spread))}
    X = f.select(FEATURE_NAMES).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (lab, i), c in zip(picks.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        w = m["w_model"].predict(w_inputs(X[i][None, :], lv)) * m["scale"]
        ax.plot(100 * lv, w, marker="o", color=c, label=f"{lab}, {days[i]}")
    allw = m["w_model"].predict(w_inputs(X, lv)).reshape(-1, len(lv)) * m["scale"]
    ax.fill_between(100 * lv, np.quantile(allw, 0.1, axis=0), np.quantile(allw, 0.9, axis=0),
                    color=INK2, alpha=0.12, label="10-90 % of all days")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("end-of-day state of charge (%)")
    ax.set_ylabel("learned value vs. ending at 50 % (EUR)")
    ax.set_title(f"Learned value of stored energy, M1 (gamma = {m['choice']['gamma']}, seed 0)", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_value_curves.png", dpi=150)
    plt.close(fig)


def cumulative_diff(res_dir, period: str, out, prefix: str) -> None:
    seeded = {m: fr for (e, m, p), fr in load_daily_seeds(res_dir / "daily", period).items() if e == "V2"}
    if "M1_learned_value" not in seeded:
        return
    m1 = mean_over_seeds(seeded["M1_learned_value"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for b in ("B3_one_step_value", "B4_rolling48h", "B0_fixed50"):
        if b not in seeded:
            continue
        j = m1.select("day", pl.col("net_adj_eur").alias("a")).join(
            mean_over_seeds(seeded[b]).select("day", pl.col("net_adj_eur").alias("b")), on="day").sort("day")
        ax.plot(j["day"].to_list(), np.cumsum((j["a"] - j["b"]).to_numpy()), color=COL[b],
                label=f"M1 - {b.split('_', 1)[0]} ({b.split('_', 1)[1].replace('_', ' ')})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_ylabel("cumulative difference (EUR, incl. terminal valuation)")
    ax.set_title(f"M1 against the baselines, {period} period (mean of seeds)", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_cumulative_diff_{period}.png", dpi=150)
    plt.close(fig)


def convergence(res_dir, out, prefix: str) -> None:
    p = res_dir / "fqi_convergence.csv"
    if not p.exists():
        return
    c = pl.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (g, t), grp in c.group_by(["gamma", "trees"], maintain_order=True):
        ax.plot(grp["iteration"], grp["mean_abs_dq"], label=f"gamma = {g}, {t}")
    ax.set_yscale("log")
    ax.set_xlabel("FQI iteration k")
    ax.set_ylabel("mean |Q_k - Q_(k-1)| (scaled)")
    ax.set_title("Fitted Q iteration convergence (training days, seed 0)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_fqi_convergence.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--period", default="test")
    add_v2_config_arg(ap)
    args = ap.parse_args()
    v2, _ = v2_config(args.config)
    prefix = v2.get("figure_prefix", "v2")
    res_dir, models_dir = v2_dir(v2, "results_dir"), v2_dir(v2, "models_dir")
    out = REPO_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    feats = pl.read_parquet(v2_dir(v2, "cache") / "features.parquet")
    import datetime as dt
    per = v2["periods"][args.period]
    end = feats["day"].max() if per["end"] == "latest" else dt.date.fromisoformat(per["end"])
    if (models_dir / "fqi_M1_seed0.pkl").exists():
        value_curves(models_dir, feats, (dt.date.fromisoformat(per["start"]), end), out, prefix, v2_battery(v2))
    for p in ("validation", args.period):
        cumulative_diff(res_dir, p, out, prefix)
    convergence(res_dir, out, prefix)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
