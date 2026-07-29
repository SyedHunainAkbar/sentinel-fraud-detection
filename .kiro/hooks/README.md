# Agent Hooks

Kiro creates hooks from natural-language descriptions in the **Agent Hooks** panel. Each
file below is the description to paste when creating that hook. They enforce quality
automatically so it is part of the workflow, not a manual afterthought.

| Hook | Trigger | Purpose |
|---|---|---|
| `on-save-test.md` | On save of `src/**/*.py` | Lint + run the module's tests |
| `on-task-complete-docs.md` | On spec task completion | Keep README/reports in sync |
| `pre-commit-pii-scan.md` | Before commit | Block any data/secret/PII leak |

Commit this folder so the hook intent is version-controlled with the project.
