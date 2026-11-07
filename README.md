# BoneTwin

BoneTwin provides research components for CT encoding, bone-density trajectory modelling, conformal risk intervals, and phenotype analysis.

## Installation

```bash
pip install -e ".[dev]"
```

## Data and execution

CT volumes, longitudinal labels, pretrained foundation-model weights, and trained checkpoints are not included. `BoneVolumes` requires caller-provided CT volumes. Training, evaluation, and triage require data and model integration and currently stop with an error.

The current CT encoders have fixed initial weights; they are not pretrained Merlin or CT-FM models. The manuscript's reported results have not been reproduced using this repository.

`sensors` contains image handling; `propagation` contains the trajectory model; `misfit` contains objectives; `alerts` contains conformal triage; `swarms` contains phenotype analysis; `residuals` and `bulletin` contain evaluation utilities. Configurations are in `configs/`.

`scripts/prepare_data.sh` lists potential data sources. Their access and usage terms apply.

```bash
pytest -q
```

This is research software and has not been validated for clinical use.
