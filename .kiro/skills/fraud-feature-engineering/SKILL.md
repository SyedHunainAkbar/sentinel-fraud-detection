---
name: fraud-feature-engineering
description: Derive leakage-free features for card-transaction fraud models — haversine customer-merchant distance, temporal features, customer age, per-category amount z-scores fit on train only, and causal per-card velocity. Use when engineering, reviewing, or extending features for the Sentinel fraud pipeline or any transaction dataset with lat/long, timestamp, and cardholder fields.
---

# Fraud Feature Engineering

Apply these patterns whenever building features from card transactions. The guiding rule
is **no leakage**: any statistic used as a feature is fit on the training split only and
applied to later data; any rolling aggregate uses only prior transactions.

## Standard feature set
1. **distance_km** — haversine distance between customer (`lat`,`long`) and merchant
   (`merch_lat`,`merch_long`). Large distances are a classic fraud signal.
2. **Temporal** — `hour`, `day_of_week`, `is_night` from `trans_date_trans_time`.
3. **age** — years between `dob` and the transaction date.
4. **log_amt** — `log1p(amt)`; amounts are heavy-tailed.
5. **amt_z_by_cat** — z-score of `amt` within each `category`. Compute mean/std on TRAIN
   only, store them, apply to test. Never recompute on the evaluation split.
6. **velocity_24h** — count of prior transactions on the same card in the trailing 24h.
   Must be causal: for row i, count only rows j with same `cc_num` and
   `unix_time_j < unix_time_i` within 86400 seconds.

## Leakage checklist (enforce on every review)
- [ ] Group statistics (`amt_z_by_cat`) fit on train, applied to test — not refit.
- [ ] Velocity uses strictly earlier timestamps (no same-instant or future rows).
- [ ] No target-derived features. No post-cutoff information in any training row.

## Scripts
- `scripts/features.py` — standalone, importable reference implementations of the
  haversine and velocity helpers with doctests. The production version lives in
  `src/sentinel/features.py`; keep the two consistent.
