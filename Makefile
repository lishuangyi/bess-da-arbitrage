# One-command reproduction. Every recipe goes through the unified CLI `python -m bess` (bess/cli.py);
# `python -m bess` alone lists all commands. PYTHON runs in the uv environment (.venv, created by `uv sync`);
# override with `make PYTHON=python ...` to use the active interpreter instead.
PYTHON ?= uv run python
BESS = $(PYTHON) -m bess

.PHONY: test data v2-cache v2 isea isea-fetch

test:
	$(PYTHON) -m pytest

data:                                ## download prices (SMARD; gaps filled from Energy-Charts after an overlap check)
	$(BESS) download

v2-cache:                            ## forward-fold XGB + reward cache (81 MILPs per day, data/v2/)
	$(BESS) cache reward --config $(V2_CONFIG)

V2_CONFIG ?= v2.yaml

v2:                                  ## from the cache: baselines, FQI selection (validation), test, tables, figures
	$(BESS) pipeline v2 --config $(V2_CONFIG)

isea:                                ## ISEA Battery Revenue Index: perfect-foresight reconciliation and comparison figure
	$(BESS) pipeline isea

isea-fetch:                          ## refresh configs/isea_monthly.csv and isea_daily.csv from the ISEA dashboard (network)
	$(BESS) isea fetch
	$(BESS) isea fetch --daily

