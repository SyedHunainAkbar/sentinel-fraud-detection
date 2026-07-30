#!/usr/bin/env bash
# Prepare a MANUALLY downloaded Kaggle archive (e.g. archive.zip) for the pipeline.
# Usage: bash scripts/prepare_local_zip.sh /path/to/archive.zip
# The extracted data lives under data/raw/ and is gitignored — it is never committed.
set -euo pipefail
ZIP="${1:?Usage: prepare_local_zip.sh /path/to/archive.zip}"
mkdir -p data/raw
unzip -o "$ZIP" -d data/raw
echo ""
echo "Extracted to data/raw/. Point the pipeline at the training file:"
echo "  export SENTINEL_DATA=data/raw/fraudTrain.csv"
echo "Then run: make train evaluate quant-risk copilot"
