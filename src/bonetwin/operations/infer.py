"""CLI entry point for triage."""

from __future__ import annotations

from pathlib import Path

import typer


def run(
    config: Path | None = typer.Argument(None),
    n: int = typer.Option(64),
    overrides: list[str] | None = typer.Argument(None),
) -> None:
    raise NotImplementedError(
        "Triage requires CT data, calibrated model weights, and data integration."
    )


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
