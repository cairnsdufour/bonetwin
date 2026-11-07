"""Config merge: schema defaults + YAML file + CLI dotlist overrides."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from bonetwin.recorder.schema import Config


def base_config() -> Config:
    cfg = OmegaConf.structured(Config)
    OmegaConf.set_readonly(cfg, False)
    return cast(Config, cfg)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = OmegaConf.structured(Config)
    OmegaConf.set_readonly(cfg, False)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.resolve(cfg)
    return cast(Config, cfg)


def to_yaml(cfg: Config) -> str:
    return OmegaConf.to_yaml(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
