#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment/main.yaml}"

torchrun --standalone --nproc_per_node=4 \
    -m bonetwin.operations.train "${CONFIG}" "${@:2}"
