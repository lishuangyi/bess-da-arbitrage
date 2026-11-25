"""BESS day-ahead scheduling research code.

Modules (each depends only on the ones above it):
  core         shared kernel: configuration, delivery-day calendar, contracts (Policy, Plan, Schedule), market rules
  input        market registry, price sources, cached price series and information sets, external index data
  forecasting  gate-closure price forecasters, their training and prediction, the multi-market interface
  milp         battery scheduling MILP, the daily and rolling MILP policies
  rl           reinforcement learning: learned end-of-day value by fitted Q iteration (value/), experiments
  output       backtesters, metrics, benchmarks, reports, figures, web data
Entry point: `python -m bess` (bess/cli.py).
"""
