"""Multi-market forecasting interface (reserved; no numerical result uses it yet).

A multi-market forecaster predicts the target market's prices from the information set of several markets at the
same decision instant (`bess.input.markets.MarketInputs`, built by `load_inputs(target, as_of_utc, exogenous)`):
e.g. neighbouring bidding zones of the coupled day-ahead auction as exogenous features.

`SingleMarketAdapter` wraps the existing single-market forecasters unchanged (same features, same numbers) and
refuses exogenous markets instead of ignoring them.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bess.forecasting.base import Forecaster
from bess.input.markets import MarketInputs


@runtime_checkable
class MarketForecaster(Protocol):
    name: str
    uses_exogenous: bool

    def predict_market(self, day, inputs: MarketInputs, resolution: str) -> np.ndarray:
        """Forecast of `inputs.target` for delivery day `day`, using only `inputs` (cut at `inputs.as_of_utc`)."""


class SingleMarketAdapter:
    """A single-market `Forecaster` seen through the multi-market interface."""

    uses_exogenous = False

    def __init__(self, forecaster: Forecaster):
        self.forecaster = forecaster
        self.name = forecaster.name

    def predict_market(self, day, inputs: MarketInputs, resolution: str) -> np.ndarray:
        if inputs.exogenous:
            raise NotImplementedError(
                f"{self.name}: single-market forecaster got exogenous markets {inputs.exogenous}; "
                "implement a MarketForecaster with uses_exogenous = True to use them")
        return self.forecaster.predict(day, inputs[inputs.target], resolution)
