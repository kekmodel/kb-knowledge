# Transaction Table Design Review

Audit date: 2026-04-21

## Why This Needs Review

The current `transactions` table is doing too many jobs:

- posted ledger entries for account, wallet, remittance, card, and box movements
- card-purchase records used as dispute targets
- pending mini or remittance transfer cases
- external payee references
- synthetic source references such as brand-coupon cashback sources

This works for final DB replay, but it weakens the meaning of DB-hash evaluation:
some records are real money movements, while others are balance-less references
that happen to be stored in the same table.

## Current Schema Mismatch

`data/kakaobank_knowledge/v0/schema/action_verifier_state.json` describes
`transactions` as a ledger-like table with required fields:

- `transaction_id`
- `customer_id`
- `source_id`
- `target_id`
- `amount`
- `currency`
- `transaction_type`
- `status`
- `posted_at`

Actual task fixtures are looser and use multiple incompatible shapes:

- `type` vs `transaction_type`
- `amount` vs `amount_krw`
- `posted_at` vs `occurred_at`, `created_at`, `sent_at`, `received_at`
- records without `customer_id`, `source_id`, `target_id`, or `posted_at`
- non-ledger `record_type` values such as `EXTERNAL_PAYEE_REFERENCE` and
  `BRAND_COUPON_CASHBACK_SOURCE`

The Pydantic DB model intentionally stores table rows as `dict[str, Any]`, so
this mismatch is not rejected.

## Current Replay Behavior

`execute_deposit_or_box_transfer` changes balances when the source or target
record has a balance-like field. It does not create a canonical transaction
ledger entry for every balance movement.

`transactions` records are sometimes updated as source records, for example
mini transfer cancellation or pending refund cases. Remittance replay creates
some transaction rows through `_upsert_transaction`, but that helper also uses
a narrower shape than the schema notes.

This means DB hash currently verifies the final balances and selected record
status updates, but it does not consistently verify that every money movement
left a normalized transaction ledger trail.

## Concrete Problem Found

`kb_manual_group_account_extra_pocket_auto_covers_withdrawal` stored
`external_payee_restaurant_019` in `transactions` as:

```json
{
  "transaction_id": "external_payee_restaurant_019",
  "record_type": "EXTERNAL_PAYEE_REFERENCE",
  "payee_name": "식당 예약금 수취처",
  "description": "모임통장 식당 예약금 외부 수취처",
  "currency": "KRW"
}
```

That made the external payee discoverable as a `target_id`, but it is not a real
transaction. The balance debit happened on the source pocket, and the target was
balance-less. Before the balance precondition fix, withdrawing from the basic
pocket before auto-moving funds could temporarily go negative and then return to
the same final balance, so DB hash did not catch the wrong order.

The immediate bug was fixed by requiring balance-backed debits to have enough
available balance. The broader design issue remains: external payees and source
references should not be indistinguishable from actual transaction ledger rows.

## Recommended V0 Direction

For the hackathon v0 scope, avoid a broad schema migration unless it is needed
for a failing evaluation. Instead:

1. Keep the existing table set for compatibility.
2. Enforce money-movement preconditions in replay:
   - no negative debit from balance-backed records
   - no lost-card compensation dispute before the lost-card report is durable
3. Treat `transactions` rows with `record_type` as references, not ledger rows.
4. Do not use action-list order matching as the primary fix for order-sensitive
   money flows.

This preserves the final DB-hash philosophy while making invalid write order
produce a failed replay or different final DB state.

## Recommended V1 Direction

For a cleaner design, split the current responsibilities:

- `transactions`: only durable ledger/event rows for actual financial or card
  events.
- `external_payees` or `counterparties`: payee references that can be transfer
  targets but do not have balances.
- `pending_transfers` or product-specific case tables: mini transfers,
  remittance pending cases, cancellation-return records, and other workflow
  objects that may later produce ledger entries.
- `transaction_sources` or explicit source fields: synthetic funding sources
  such as coupon cashback sources, if they need to remain DB-visible.

If `transactions` remains a union table, every row should at least include a
stable discriminator such as `record_type`, and replay helpers should branch on
that discriminator instead of inferring behavior from loose field presence.

## Open Design Question

Should every balance-changing write automatically create a canonical transaction
ledger row?

Pros:

- DB hash verifies an auditable money-movement trail, not only balances.
- External payments and internal transfers become easier to inspect.

Cons:

- Deterministic generated transaction IDs must be defined so the model is not
  asked to guess hidden IDs.
- If sequence fields are included too broadly, DB hash may start evaluating
  harmless commutative order differences.

For v0, the safer answer is no broad automatic ledger generation yet. Add it
only for task families where the transaction record itself is part of the
customer-visible outcome.
