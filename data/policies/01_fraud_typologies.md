# Fraud Typologies Reference

## Card-not-present (CNP) testing
Fraudsters validate stolen card numbers with many small online transactions in a short
window before a large purchase. Signal: high transaction velocity on a card, small
amounts followed by a spike, online (net) categories.

## Geographic impossibility
A physical-present transaction far from the customer's home location, or two transactions
whose locations could not be reached in the elapsed time. Signal: large customer-to-
merchant distance, especially combined with a new merchant.

## Account takeover
After credential compromise, spending pattern shifts abruptly: new categories, higher
amounts, unusual hours. Signal: amount z-score far above the customer's category norm and
night-time activity.

## High-value single-shot
A single large purchase inconsistent with the customer's history, often in shopping_net or
travel categories. Signal: amount in the far right tail for the category.
