"""Forecast accuracy (MAE / RMSE) under the same gate-closure information rule as the backtest."""
from __future__ import annotations

import numpy as np

from bess.data import calendar as cal
from bess.data.smard import MarketData


def forecast_errors(forecaster, data: MarketData, days: list, resolution: str, gate_closure_hhmm: str = "12:00",
                    publication_hhmm: str = "13:00") -> dict:
    series = data.series(resolution)
    err = []
    for d in days:
        info = data.known_at(cal.gate_closure_utc(d, gate_closure_hhmm), publication_hhmm)
        f = forecaster.predict(d, info, resolution)
        err.append(f - series.day_prices(d))
    e = np.concatenate(err)
    return {"n_days": len(days), "n_points": int(len(e)), "mae_eur_mwh": float(np.mean(np.abs(e))),
            "rmse_eur_mwh": float(np.sqrt(np.mean(e ** 2))), "bias_eur_mwh": float(np.mean(e))}
