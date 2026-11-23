"""Configuration loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(name: str | Path) -> dict:
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Battery:
    p_max_mw: float = 1.0
    e_max_mwh: float = 2.0
    eta_ch: float = 0.9 ** 0.5
    eta_dis: float = 0.9 ** 0.5
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.50
    soc_final: float = 0.50
    c_deg_eur_per_mwh: float = 10.0
    # Optional throughput limit: discharged energy per delivery day <= max_cycles_per_day * e_max_mwh.
    # None = unlimited (the project default). Used to match external benchmarks (ISEA index).
    max_cycles_per_day: float | None = None
    # What the daily cycle limit counts: "discharge" = energy delivered to the grid (p_dis * dt);
    # "stored_charge" = energy added to the store (eta_ch * p_ch * dt), as in the ISEA index code.
    cycle_basis: str = "discharge"

    @property
    def e_min(self) -> float:
        return self.soc_min * self.e_max_mwh

    @property
    def e_upper(self) -> float:
        return self.soc_max * self.e_max_mwh

    @property
    def e_init(self) -> float:
        return self.soc_init * self.e_max_mwh

    @property
    def e_final(self) -> float:
        return self.soc_final * self.e_max_mwh

    @classmethod
    def from_yaml(cls, name: str | Path = "battery.yaml") -> "Battery":
        return cls(**load_yaml(name))


def experiment_config(name: str | Path = "experiment.yaml") -> dict:
    return load_yaml(name)


def results_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["backtest"]["results_dir"]


def models_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg.get("models_dir", "models")


def protocol_name(cfg: dict) -> str:
    return cfg.get("protocol", "brief")


def add_config_arg(ap) -> None:
    ap.add_argument("--config", default="experiment.yaml",
                    help="experiment config in configs/ (data split protocol), default experiment.yaml")
