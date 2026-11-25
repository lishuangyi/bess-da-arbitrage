"""Unified CLI, market registry, price sources and the multi-market interfaces."""
import dataclasses
import datetime as dt
import importlib

import numpy as np
import pytest

from bess import cli
from bess.core import calendar as cal
from bess.core.config import experiment_config, v2_config
from bess.forecasting.multimarket import MarketForecaster, SingleMarketAdapter
from bess.forecasting.naive import SeasonalNaive
from bess.input.markets import (
    MarketInputs,
    get_market,
    load_inputs,
    registry,
    resolve_market,
)
from bess.input.sources.base import get_source, registered_sources


def test_registry_keeps_the_previous_data_settings():
    de = get_market("de_lu")
    assert de.source == {"kind": "smard", "base_url": "https://www.smard.de/app/chart_data", "filter_id": 4169,
                         "region": "DE-LU"}
    assert (de.cache_dir, de.qh_market_start, de.download_start) == ("data", "2025-10-01", "2019-12-01")
    assert (de.gate_closure_local, de.publication_local, de.timezone) == ("12:00", "13:00", "Europe/Berlin")
    at = get_market("at")
    assert (at.source["filter_id"], at.source["region"], at.cache_dir) == (4170, "DE-LU", "data/at")


def test_configs_resolve_to_their_market():
    assert resolve_market(experiment_config("experiment.yaml")).id == "de_lu"
    assert resolve_market(experiment_config("experiment_at.yaml")).id == "at"
    v2, base = v2_config("v2_isea.yaml")
    assert resolve_market(v2).id == "de_lu" and resolve_market(base).id == "de_lu"
    for name in ("experiment.yaml", "experiment_at.yaml"):
        cfg = experiment_config(name)
        assert "data" not in cfg and "day_ahead" not in cfg, name


def test_day_ahead_market_from_spec():
    m = get_market("de_lu").day_ahead_market()
    d = dt.date(2026, 3, 29)   # DST day
    assert m.gate_closure_utc(d) == cal.gate_closure_utc(d, "12:00") == get_market("de_lu").gate_closure_utc(d)


def test_unsupported_markets_fail_loudly():
    de = get_market("de_lu")
    with pytest.raises(NotImplementedError, match="timezone"):
        resolve_market(dataclasses.replace(de, id="gb", timezone="Europe/London"))
    with pytest.raises(NotImplementedError, match="holiday"):
        resolve_market(dataclasses.replace(de, id="fr", holidays="FR"))
    with pytest.raises(KeyError):
        get_market("nowhere")
    assert {"smard", "entsoe"} <= set(registered_sources())
    with pytest.raises(NotImplementedError, match="ENTSO-E"):
        get_source("entsoe").fetch(dataclasses.replace(de, id="fr", source={"kind": "entsoe"}), None, "hour")
    assert set(registry()) >= {"de_lu", "at"}


def test_load_inputs_cuts_every_market_at_the_same_instant(mdata):
    day = dt.date(2025, 11, 12)
    gc = cal.gate_closure_utc(day)
    with pytest.raises(ValueError):
        load_inputs("de_lu", gc.replace(tzinfo=None), data={"de_lu": mdata})
    inp = load_inputs("de_lu", gc, exogenous=["at"], data={"de_lu": mdata, "at": mdata})
    assert inp.exogenous == ["at"]
    for mid in ("de_lu", "at"):
        assert inp[mid].latest_timestamp() < cal.day_start_utc(day)
        assert inp[mid].latest_timestamp() >= cal.day_start_utc(day - dt.timedelta(days=1))


def test_single_market_adapter_matches_the_forecaster_and_refuses_exogenous(mdata):
    day = dt.date(2025, 11, 12)
    gc = cal.gate_closure_utc(day)
    fc = SingleMarketAdapter(SeasonalNaive())
    assert isinstance(fc, MarketForecaster)
    own = load_inputs("de_lu", gc, data={"de_lu": mdata})
    np.testing.assert_array_equal(fc.predict_market(day, own, "quarterhour"),
                                  SeasonalNaive().predict(day, mdata.known_at(gc), "quarterhour"))
    both = MarketInputs("de_lu", gc, {"de_lu": own["de_lu"], "at": own["de_lu"]})
    with pytest.raises(NotImplementedError):
        fc.predict_market(day, both, "quarterhour")


@pytest.mark.parametrize("key", cli.available_commands())
def test_every_command_resolves_and_has_help(key, capsys):
    for mod in cli.COMMANDS[key][0]:
        if mod != "bess.cli":
            assert callable(importlib.import_module(mod).main), mod
    tokens = [key[0]] + ([key[1]] if key[1] else []) + ["--help"]
    if key == ("markets", None):
        cli.run_command(["markets"])
        assert "de_lu" in capsys.readouterr().out
        return
    with pytest.raises(SystemExit) as e:
        cli.run_command(tokens)
    assert e.value.code == 0
    assert "usage: python -m bess" in capsys.readouterr().out


def test_unknown_commands_and_pipelines(capsys):
    with pytest.raises(SystemExit, match="sub-commands of 'train'"):
        cli.run_command(["train"])
    with pytest.raises(SystemExit, match="unknown command"):
        cli.run_command(["frobnicate"])
    cli.main(["pipeline", "v2", "--config", "v2_isea.yaml", "--dry-run"])
    out = capsys.readouterr().out
    assert "python -m bess train fqi --config v2_isea.yaml" in out
    cli.main(["pipeline", "data", "--config", "experiment_at.yaml", "--dry-run"])
    assert "python -m bess download --config experiment_at.yaml" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="takes no --config"):
        cli.main(["pipeline", "isea", "--config", "v2.yaml", "--dry-run"])
    for name, (_, steps) in cli.available_pipelines().items():
        for tokens, _ in steps:
            sub = tokens[1] if len(tokens) > 1 and (tokens[0], tokens[1]) in cli.COMMANDS else None
            assert (tokens[0], sub) in cli.COMMANDS, (name, tokens)


def test_commands_of_missing_packages_are_hidden(monkeypatch, capsys):
    monkeypatch.setitem(cli.COMMANDS, ("experiment", "absent"), (("bess.no_such_package.module",), "not here"))
    assert ("experiment", "absent") not in cli.available_commands()
    cli.main([])
    assert "absent" not in capsys.readouterr().out
