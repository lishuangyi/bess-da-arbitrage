"""Forecast-based scheduling against the ISEA Battery Revenue Index: cumulative revenue on the same delivery days.

Usage: python -m bess figures isea [--config v2_isea.yaml]
Writes results/figures/<prefix>_vs_isea_index.png: one column per period (validation 2025, test 2026); top row the
cumulative revenue in kEUR per MW of the ISEA day-ahead index (hindsight, the profile's `isea_reference` series of
configs/isea_daily.csv) and of our methods, bottom row the monthly share of the index. Days are those with both an
ISEA value and a result of every plotted method; ISEA's zero-revenue days are kept as published (the "literal"
variant of <results_dir>/isea_share_<period>.csv); our daily values include the terminal valuation (net_adj_eur),
learned-value rows are the mean over seeds.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from bess.core.config import REPO_ROOT, add_v2_config_arg, v2_battery, v2_config, v2_dir  # noqa: E402
from bess.output.io import load_daily_seeds, mean_over_seeds  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# forecast-based methods in a fixed hue order (validated for colour-vision deficiency on SURFACE)
SERIES = [("M1_learned_value", "MILP + FQI value (main method)", "#2a78d6"),
          ("B4_rolling48h", "rolling 48 h MILP", "#eb6834"),
          ("B0_fixed50", "daily MILP, end fixed at 50 %", "#1baf7a")]
PF = ("PF_fixed50", "perfect foresight (backtester check)", INK2)
PERIOD_TITLE = {"validation": "Validation 2025", "test": "Test 2026 (retrospective)"}


def isea_daily(ref: dict) -> pl.DataFrame:
    path = REPO_ROOT / "configs" / "isea_daily.csv"
    return (pl.read_csv(path, try_parse_dates=True)
            .filter((pl.col("system_h") == ref["system_h"]) & (pl.col("cycles") == ref["cycles"]))
            .select(pl.col("day").cast(pl.Date), pl.col("eur_per_mw_day").alias("isea")))


def matched(res_dir, period: str, isea: pl.DataFrame, p_max_mw: float) -> pl.DataFrame | None:
    """Per day: ISEA EUR/MW and every plotted method's EUR/MW, on the days all of them cover."""
    seeded = {m: fr for (e, m, p), fr in load_daily_seeds(res_dir / "daily", period).items() if e == "V2"}
    methods = [m for m, _, _ in SERIES + [PF] if m in seeded]
    if "M1_learned_value" not in methods:
        return None
    j = isea
    for m in methods:
        ours = mean_over_seeds(seeded[m]).select("day", (pl.col("net_adj_eur") / p_max_mw).alias(m))
        j = j.join(ours, on="day", how="inner")
    return j.sort("day") if j.height else None


def draw(ax, ax2, j: pl.DataFrame, title: str) -> None:
    days = j["day"].to_list()
    total = float(j["isea"].sum())
    ax.plot(days, np.cumsum(j["isea"].to_numpy()) / 1e3, color=INK, lw=2.4, label="ISEA day-ahead index (hindsight) = 100 %")
    for m, name, col in [PF] + SERIES:
        if m not in j.columns:
            continue
        y = np.cumsum(j[m].to_numpy()) / 1e3
        ax.plot(days, y, color=col, lw=2.0 if m != PF[0] else 1.4, ls="--" if m == PF[0] else "-",
                label=f"{name}: {100 * y[-1] * 1e3 / total:.1f} %")
    ax.set_title(f"{title}, {j.height} days", loc="left", fontsize=10.5)
    ax.set_ylabel("cumulative revenue (kEUR per MW)")
    ax.legend(fontsize=8, loc="upper left")
    mon = (j.with_columns(pl.col("day").dt.truncate("1mo").alias("month"))
           .group_by("month").agg(pl.exclude("day").sum()).sort("month"))
    ax2.axhline(100, color=INK, lw=1.4)
    for m, _, col in SERIES:
        if m in mon.columns:
            ax2.plot(mon["month"].to_list(), (100 * mon[m] / mon["isea"]).to_numpy(), color=col, lw=2.0, marker="o", ms=4)
    ax2.set_ylabel("monthly share of index (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def isea_comparison(v2: dict, out_dir) -> list:
    ref = v2.get("isea_reference")
    if not ref or not (REPO_ROOT / "configs" / "isea_daily.csv").exists():
        return []
    res_dir, p_max = v2_dir(v2, "results_dir"), v2_battery(v2).p_max_mw
    isea = isea_daily(ref)
    panels = [(p, j) for p in ("validation", "test") if (j := matched(res_dir, p, isea, p_max)) is not None]
    if not panels:
        return []
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, len(panels), figsize=(6.4 * len(panels), 6.6), height_ratios=[2.3, 1],
                             sharex="col", sharey="row", squeeze=False)
    for c, (p, j) in enumerate(panels):
        draw(axes[0, c], axes[1, c], j, PERIOD_TITLE.get(p, p))
        if c > 0:
            axes[0, c].set_ylabel("")
            axes[1, c].set_ylabel("")
    fig.suptitle(f"Gate-closure forecasts vs the ISEA Battery Revenue Index (1 MW / {ref['system_h']} h, "
                 f"{ref['cycles']} cycles per day)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    path = out_dir / f"{v2.get('figure_prefix', 'v2')}_vs_isea_index.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_v2_config_arg(ap)
    ap.set_defaults(config="v2_isea.yaml")
    args = ap.parse_args(argv)
    v2, _ = v2_config(args.config)
    out = REPO_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = isea_comparison(v2, out)
    print(f"figure -> {paths[0]}" if paths else f"{args.config}: no ISEA reference or no matched results - skipped")


if __name__ == "__main__":
    main()
