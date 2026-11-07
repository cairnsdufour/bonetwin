#!/usr/bin/env bash
set -euo pipefail

# Fetch instructions for the eight public spine-CT corpora. MrOS and SOF are
# access-gated by a Data Use Agreement and must be requested through their
# portals; the remaining six are directly downloadable.

DEST="${1:-data/spine}"
mkdir -p "${DEST}"

cat <<'EOF'
VerSe            https://github.com/anjany/verse
CT-RATE          https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
NLST             https://www.cancerimagingarchive.net/collection/nlst/
MrOS             https://mrosonline.ucsf.edu/          (registration + DUA)
SOF              https://sofonline.ucsf.edu/           (registration + DUA)
LUMOS            https://keyueshi.github.io/LUMOS/
TotalSegmentator https://github.com/wasserth/TotalSegmentator   (Zenodo 6802614)
CTSpine1K        https://github.com/MIRACLE-Center/CTSpine1K
EOF

echo "A loader for harmonised vertebra CT volumes and longitudinal T-score labels must be implemented."
