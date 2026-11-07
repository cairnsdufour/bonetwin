"""CLI: export the transition decoder h_psi to ONNX."""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from bonetwin.propagation.picker import TransitionDecoder
from bonetwin.recorder.config import load_config
from bonetwin.recorder.telemetry import get_logger

_LOG = get_logger("bonetwin.export")


def run(
    config: Path | None = typer.Argument(None),
    out: Path = typer.Option(Path("outputs/decoder.onnx")),
    overrides: list[str] | None = typer.Argument(None),
) -> None:
    cfg = load_config(config, list(overrides) if overrides else None)
    decoder = TransitionDecoder(
        cfg.propagation.latent_dim, cfg.propagation.decoder_width, cfg.propagation.decoder_depth
    ).eval()
    example_input = torch.zeros(1, cfg.propagation.latent_dim)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        decoder,
        (example_input,),
        str(out),
        input_names=["latent"],
        output_names=["tscore"],
        dynamic_axes={"latent": {0: "batch"}, "tscore": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    _LOG.info("exported decoder to %s", out)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
