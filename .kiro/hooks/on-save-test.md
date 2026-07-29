# Hook: on-save test & lint

**Trigger:** When a file matching `src/**/*.py` is saved.

**Action (paste as the hook prompt):**
A Python source file under `src/` was just saved. Run `ruff check` on it and fix trivial
lint issues. Then run the matching test file in `tests/` (e.g., `src/sentinel/features.py`
-> `tests/test_features.py`) with pytest. If tests fail, summarize the failure and propose
a minimal fix. Do not modify unrelated files. Keep changes small and leakage-free per the
risk.md steering.
