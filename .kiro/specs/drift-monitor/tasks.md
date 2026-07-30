# Tasks — Production Drift Monitor

- [x] 1. Add drift config constants to `config.py` (PSI thresholds, degradation ratio, n_windows)
- [x] 2. Persist `X_train` numeric features in `train.py` artifacts for reference distributions
- [x] 3. Create `src/sentinel/drift/__init__.py` package marker
- [x] 4. Implement `src/sentinel/drift/drift_psi.py` (compute_feature_psi, compute_score_psi, classify_severity)
- [x] 5. Implement `src/sentinel/drift/drift_performance.py` (rolling_window_metrics, detect_degradation)
- [x] 6. Implement `src/sentinel/drift/drift_monitor.py` orchestrator with `run_drift_monitor()` and `main()`
- [x] 7. Add `make drift` target to the Makefile
- [x] 8. Write `tests/test_drift_psi.py` (PSI severity classification, stable/shifted distributions)
- [x] 9. Write `tests/test_drift_performance.py` (rolling metrics shape, degradation detection)
- [x] 10. Run `make test lint` and fix any issues
