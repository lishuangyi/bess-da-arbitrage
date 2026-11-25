# One-command reproduction. Every recipe goes through the unified CLI `python -m bess` (bess/cli.py);
# `python -m bess` alone lists all commands. PYTHON defaults to the active interpreter.
PYTHON ?= python
BESS = $(PYTHON) -m bess

.PHONY: test data v2-cache v2 isea isea-fetch diagrams

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

DIAGRAM_SKILL ?= $(HOME)/.claude/plugins/cache/diagram-design/diagram-design/2.6.164/skills/diagram-design

diagrams:                            ## diagram-design sources -> docs/diagrams/*.svg (system fonts)
	$(PYTHON) docs/diagrams/build_diagrams.py
	for f in timing timing-dark scheme scheme-dark scheme-zh scheme-zh-dark; do \
	  python3 $(DIAGRAM_SKILL)/scripts/export_svg.py docs/diagrams/$$f.html docs/diagrams/$$f.svg --system-fonts; done

