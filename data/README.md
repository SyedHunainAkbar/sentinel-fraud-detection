# Data

The real dataset is **not** committed (it is ~350 MB and exceeds GitHub limits).

- **Sample** (`sample/transactions_sample.csv`): a tiny synthetic file with the Sparkov
  schema so tests and CI run offline. Regenerate with `make sample`.
- **Full dataset**: [Sparkov Credit Card Transactions](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
  Fetch with `make data` (requires the Kaggle CLI), then:
  `export SENTINEL_DATA=data/raw/fraudTrain.csv`.

Never commit anything under `data/` other than `sample/` and this README.
