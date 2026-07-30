#!/usr/bin/env bash
# Download the full Sparkov dataset from Kaggle. Requires the Kaggle CLI configured
# (~/.kaggle/kaggle.json). The data is gitignored and must never be committed.
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
echo "Downloaded to data/raw/. Point the pipeline at it with:"
echo "  export SENTINEL_DATA=data/raw/fraudTrain.csv"
