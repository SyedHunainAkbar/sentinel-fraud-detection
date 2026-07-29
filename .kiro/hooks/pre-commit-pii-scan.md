# Hook: pre-commit PII / secret scan

**Trigger:** Before a git commit.

**Action (paste as the hook prompt):**
Before committing, scan staged changes for problems and BLOCK the commit if any are found:
1. Any file under `data/` other than `data/sample/` (raw dataset must never be committed).
2. Files larger than 5 MB.
3. Anything resembling credentials or PII: API keys, tokens, `kaggle.json`, real card
   numbers (long digit runs), emails, or `.env` contents.
If clean, allow the commit. If not, list the offending files and the reason, and suggest
adding them to `.gitignore`.
