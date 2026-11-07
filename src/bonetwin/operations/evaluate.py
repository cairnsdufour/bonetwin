"""CLI entry point for evaluation."""

from __future__ import annotations

from pathlib import Path

import typer


def run(
    config: Path | None = typer.Argument(None),
    overrides: list[str] | None = typer.Argument(None),
) -> None:
    raise NotImplementedError(
        "Evaluation requires independent CT data and trained model weights."
    )


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
