# Investigation Playbook

## Triage steps
1. Confirm the model score and the top contributing features.
2. Pull the customer's recent transaction history; assess whether the flagged transaction
   is consistent with established behavior.
3. Match the pattern to a known typology (see typologies reference).
4. Check the transaction amount against the customer's category norm.

## Disposition guidance
- **Escalate** when the score is high AND the pattern matches a typology AND the amount is
  material relative to the customer's norm.
- **Request info** when signals are mixed or the customer history is sparse; contact the
  customer to verify before adverse action.
- **Clear** when the transaction is consistent with history and no typology matches, even
  if the score is moderately elevated.

## Human-in-the-loop
The model and copilot recommend; a human analyst makes the final disposition. Never take
automated adverse action on a single score.
