"""CLI: train the BoneTwin digital twin."""

from __future__ import annotations

from pathlib import Path

import typer

from bonetwin.inversion.pipeline import BoneTwinPipeline
from bonetwin.recorder.config import load_config, save_config
from bonetwin.recorder.telemetry import get_logger

_LOG = get_logger("bonetwin.train")


def run(
    config: Path | None = typer.Argument(None),
    overrides: list[str] | None = typer.Argument(None),
) -> None:
    cfg = load_config(config, list(overrides) if overrides else None)
    summary = BoneTwinPipeline(cfg).run()
    save_config(cfg, Path(cfg.out_dir) / "resolved.yaml")
    _LOG.info("training summary: %s", summary)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
