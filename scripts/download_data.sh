#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download the PlantVillage dataset into data/raw/
#
# Two methods are provided:
#   A) KAGGLE  -> full dataset, official source (needs a free Kaggle account)
#   B) GITHUB  -> full dataset OR a subset of classes (no account needed)
#
# Usage:
#   bash scripts/download_data.sh kaggle        # full set via Kaggle
#   bash scripts/download_data.sh github-full   # full set via GitHub mirror
#   bash scripts/download_data.sh github-subset # only the classes listed below
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # project root
mkdir -p data/raw
METHOD="${1:-github-subset}"

# Classes to pull for the "github-subset" method (edit freely):
SUBSET=(
  "Apple___Apple_scab"
  "Apple___Black_rot"
  "Apple___healthy"
  "Potato___Early_blight"
  "Potato___Late_blight"
  "Potato___healthy"
)

case "$METHOD" in
  kaggle)
    # Requires ~/.kaggle/kaggle.json (Account -> Create New API Token)
    echo ">> Downloading full PlantVillage set from Kaggle..."
    kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data --unzip
    # Kaggle unzips to data/plantvillage dataset/... -> move the color set into data/raw
    echo ">> Move the 'color' folder contents into data/raw/ (path varies by mirror)."
    ;;

  github-full)
    echo ">> Cloning full color set from GitHub mirror (~2 GB)..."
    git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git .pv_tmp
    mv .pv_tmp/raw/color/* data/raw/
    rm -rf .pv_tmp
    ;;

  github-subset)
    echo ">> Sparse-checkout of ${#SUBSET[@]} classes from GitHub mirror..."
    rm -rf .pv_tmp
    git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/spMohanty/PlantVillage-Dataset.git .pv_tmp
    ( cd .pv_tmp
      git sparse-checkout init --cone
      paths=(); for c in "${SUBSET[@]}"; do paths+=("raw/color/$c"); done
      git sparse-checkout set "${paths[@]}"
      git checkout )
    mv .pv_tmp/raw/color/* data/raw/
    rm -rf .pv_tmp
    ;;

  *)
    echo "Unknown method: $METHOD"; echo "Use: kaggle | github-full | github-subset"; exit 1;;
esac

echo ">> Done. Classes now in data/raw/:"
ls data/raw
echo ">> Total images: $(find data/raw -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
