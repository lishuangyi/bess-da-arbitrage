"""Unified command line: `python -m bess <command> [<sub-command>] [options]` (or the `bess` console script).

Every command is a `main(argv)` function in the module that owns it, imported only when the command runs; the same
options as before are accepted (`python -m bess <command> --help` lists them). Pipelines chain commands in the
order of the Makefile targets.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time

# (command, sub-command) -> (module(s) with main(argv), one-line help). Sub-command None = command without one.
COMMANDS: dict[tuple[str, str | None], tuple[tuple[str, ...], str]] = {
    # input
    ("download", None): (("bess.input.download",), "download and cache the prices of a market (SMARD)"),
    ("markets", None): (("bess.cli",), "list the markets of configs/markets.yaml and their status"),
    ("isea", "fetch"): (("bess.input.isea",), "fetch the ISEA Battery Revenue Index (monthly / --daily)"),
    # forecasting
    ("forecast", None): (("bess.forecasting.predict",), "gate-closure price forecast of one delivery day"),
    ("train", "xgb"): (("bess.forecasting.train",), "fit the XGBoost forecasters (forward-fold or single)"),
    # rl
    ("train", "fqi"): (("bess.rl.value.train",), "fitted Q iteration: learned end-of-day SoC value (v2)"),
    ("cache", "reward"): (("bess.rl.value.build_cache",), "v2 reward cache: (start, end) SoC MILPs per day"),
    ("cache", "cap"): (("bess.rl.experiments.cap_cache",), "v3 cap cache: MILPs per daily cycle cap"),
    ("experiment", "budget"): (("bess.rl.experiments.budget_run",), "v3 cycle-quota experiment (one fold)"),
    # milp + output: backtests and their evaluation
    ("backtest", "chain"): (("bess.output.run_chain",), "v2 chained backtests (SoC carried across days)"),
    ("benchmark", "v2"): (("bess.output.v2_summary",), "v2 summary, paired differences, share of the ISEA index"),
    ("benchmark", "public"): (("bess.output.public_benchmarks",), "re-run under ISEA / public benchmark assumptions"),
    ("benchmark", "headroom"): (("bess.output.headroom",), "hindsight headroom of end-of-day policies"),
    ("figures", "v2"): (("bess.output.figures.v2",), "v2 figures (value curves, differences, ISEA comparison)"),
    ("figures", "isea"): (("bess.output.figures.isea",), "forecast-based methods vs the ISEA index (README figure)"),
    ("web", "scheme"): (("bess.output.web.scheme",), "web/data.js for web/index.html"),
    ("web", "isea"): (("bess.output.web.isea",), "web/isea_data.js for web/isea.html"),
}

# Named workflows (same order as the Makefile targets). Steps tagged "cfg" receive the pipeline's --config.
PIPELINES: dict[str, tuple[str, list[tuple[list[str], bool]]]] = {
    "data": ("download the prices of the config's market (--config experiment_at.yaml for Austria)",
             [(["download"], True)]),
    "v2": ("v2 from the reward cache: baselines, FQI selection, test, tables, figures (--config v2_isea.yaml for "
           "the ISEA battery)", [
        (["backtest", "chain", "--period", "validation", "--methods", "pf,b0,b1,b2,b4"], True),
        (["train", "fqi"], True),
        (["benchmark", "v2", "--period", "validation"], True),
        (["backtest", "chain", "--period", "test", "--methods", "pf,b0,b1,b2,b4,b3,m1,m2"], True),
        (["benchmark", "v2", "--period", "test"], True),
        (["figures", "v2", "--period", "test"], True)]),
    "v3": ("v3 throughput-limited battery: cap cache, then the weekly / monthly / annual quota folds 2024 and 2025", [
        (["cache", "cap"], False)] + [
        (["experiment", "budget", "--kind", kind, "--cycles", cyc, "--fold", fold, "--seeds", "0"], False)
        for fold in ("2024", "2025") for kind, cyc in (("week", "5"), ("month", "0.7"), ("year", "255"))]),
    "isea": ("ISEA index: reconciliation of perfect foresight, comparison figure, web page data", [
        (["benchmark", "public"], False),
        (["figures", "isea"], False),
        (["web", "isea"], False)]),
    "web": ("rebuild both web pages' data", [(["web", "scheme"], False), (["web", "isea"], False)]),
}


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:          # a parent package is not part of this checkout
        return False


def _available(key) -> bool:
    return all(m == "bess.cli" or _module_present(m) for m in COMMANDS[key][0])


def available_commands() -> list:
    return [k for k in COMMANDS if _available(k)]


def available_pipelines() -> dict:
    def ok(tokens):
        sub = tokens[1] if len(tokens) > 1 and (tokens[0], tokens[1]) in COMMANDS else None
        return (tokens[0], sub) in COMMANDS and _available((tokens[0], sub))
    return {n: v for n, v in PIPELINES.items() if any(ok(tok) for tok, _ in v[1])}


def _step_available(tokens: list[str]) -> bool:
    sub = tokens[1] if len(tokens) > 1 and (tokens[0], tokens[1]) in COMMANDS else None
    return (tokens[0], sub) in COMMANDS and _available((tokens[0], sub))


def _usage() -> str:
    lines = ["usage: python -m bess <command> [<sub-command>] [options]", "",
             "Commands (python -m bess <command> [<sub-command>] --help for the options):"]
    for cmd, sub in available_commands():
        hlp = COMMANDS[(cmd, sub)][1]
        lines.append(f"  {(cmd + ' ' + (sub or '')).strip():<26}{hlp}")
    lines.append(f"  {'pipeline <name>':<26}run a named workflow: {', '.join(available_pipelines())}")
    return "\n".join(lines)


def list_markets(argv: list[str] | None = None) -> None:
    from bess.input.markets import _check_supported, registry
    from bess.input.sources.base import registered_sources

    for mid, spec in registry().items():
        try:
            _check_supported(spec)
            status = "ok"
        except NotImplementedError as e:
            status = f"not supported: {e}"
        cached = (spec.cache_path / "prices_hour.parquet").exists()
        print(f"{mid:<8}{spec.name:<42}source={spec.source.get('kind')}  cache={spec.cache_dir} "
              f"({'present' if cached else 'missing'})  {status}")
    print(f"price sources: {', '.join(registered_sources())}")


def run_command(tokens: list[str]) -> None:
    """Run one command given as command-line tokens, e.g. ["backtest", "chain", "--period", "test"]."""
    if not tokens:
        raise SystemExit(_usage())
    cmd = tokens[0]
    sub = tokens[1] if len(tokens) > 1 and (cmd, tokens[1]) in COMMANDS else None
    key = (cmd, sub)
    if key not in COMMANDS or not _available(key):
        subs = [s for (c, s) in available_commands() if c == cmd and s]
        hint = f"; sub-commands of {cmd!r}: {', '.join(subs)}" if subs else ""
        raise SystemExit(f"unknown command {' '.join(tokens[:2])!r}{hint}\n\n{_usage()}")
    rest = tokens[2:] if sub else tokens[1:]
    prog, sys.argv[0] = sys.argv[0], f"python -m bess {cmd}" + (f" {sub}" if sub else "")   # argparse usage line
    try:
        _run_modules(COMMANDS[key][0], rest)
    finally:
        sys.argv[0] = prog


def _run_modules(modules: tuple[str, ...], rest: list[str]) -> None:
    for mod in modules:
        if mod == "bess.cli":
            list_markets(rest)
            continue
        importlib.import_module(mod).main(rest)


def run_pipeline(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="bess pipeline", description="run a named workflow")
    ap.add_argument("name", choices=list(available_pipelines()))
    ap.add_argument("--config", default=None, help="profile passed to the steps that take one (e.g. v2_isea.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="print the steps only")
    args = ap.parse_args(argv)
    steps = PIPELINES[args.name][1]
    if args.config and not any(takes_cfg for _, takes_cfg in steps):
        raise SystemExit(f"pipeline {args.name!r} takes no --config")
    for tokens, takes_cfg in steps:
        step = tokens + (["--config", args.config] if takes_cfg and args.config else [])
        if not _step_available(tokens):
            print(f"\n--- skipped (not part of this checkout): python -m bess {' '.join(step)}", flush=True)
            continue
        print(f"\n>>> python -m bess {' '.join(step)}", flush=True)
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        run_command(step)
        print(f"<<< done in {time.perf_counter() - t0:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return
    if argv[0] == "pipeline":
        run_pipeline(argv[1:])
        return
    run_command(argv)
