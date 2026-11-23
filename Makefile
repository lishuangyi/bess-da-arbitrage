# One-command reproduction. PYTHON defaults to the active interpreter.
PYTHON ?= python

.PHONY: data test v2-cache v2 isea isea-fetch

data:                                ## download SMARD prices
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest

v2-cache:                            ## forward-fold XGB + reward cache (81 MILPs per day, data/v2/)
	$(PYTHON) scripts/build_reward_cache.py

v2:                                  ## from the cache: baselines, FQI selection (validation), test, tables, figures
	$(PYTHON) scripts/run_chain.py --period validation --methods pf,b0,b1,b2,b4
	$(PYTHON) scripts/train_fqi.py
	$(PYTHON) scripts/v2_summary.py --period validation
	$(PYTHON) scripts/run_chain.py --period test --methods pf,b0,b1,b2,b4,b3,m1,m2
	$(PYTHON) scripts/v2_summary.py --period test
	$(PYTHON) scripts/v2_figures.py --period test

isea:                                ## ISEA Battery Revenue Index: perfect-foresight reconciliation
	$(PYTHON) scripts/compare_public_benchmarks.py

isea-fetch:                          ## refresh configs/isea_monthly.csv and isea_daily.csv from the ISEA dashboard (network)
	$(PYTHON) scripts/fetch_isea_index.py
	$(PYTHON) scripts/fetch_isea_index.py --daily
