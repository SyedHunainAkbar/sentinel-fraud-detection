# Hook: keep docs in sync on task completion

**Trigger:** When a task in `.kiro/specs/fraud-detection/tasks.md` is marked complete.

**Action (paste as the hook prompt):**
A spec task was just completed. Update the README results tables and any affected file in
`reports/` so documentation reflects the current code and metrics. If `reports/evaluation.json`
exists, regenerate the model card via the `model-card` skill. Never invent numbers — read
them from `reports/evaluation.json`. Keep the executive summary's headline as a dollar figure.
