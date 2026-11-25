"""Configuration loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/bess/core/config.py -> repository root
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


# --------------------------------------------------------------------------- v2 profiles (configs/v2*.yaml)

def v2_config(name: str = "v2.yaml") -> tuple[dict, dict]:
    """(v2 profile, its base experiment config)."""
    v2 = load_yaml(name)
    return v2, experiment_config(v2["base_config"])


def v2_battery(v2: dict) -> Battery:
    return Battery.from_yaml(v2.get("battery", "battery.yaml"))


def forecaster_path(v2: dict) -> Path:
    return REPO_ROOT / v2.get("forecaster", "models/v2/xgb_forward_fold.pkl")


def add_v2_config_arg(ap) -> None:
    ap.add_argument("--config", default="v2.yaml",
                    help="v2 profile in configs/: v2.yaml (project battery) or v2_isea.yaml (ISEA index battery)")


def v2_dir(v2: dict, key: str) -> Path:
    return REPO_ROOT / (v2[key] if key != "cache" else v2["cache"]["dir"])
