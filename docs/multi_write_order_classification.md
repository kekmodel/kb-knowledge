# Multi-Write Order Classification

Audit date: 2026-04-21

This note classifies v0 KakaoBank tasks with more than one assistant-side
state-changing write action. The criterion is not whether the expected action
list has an arbitrary order, but whether a wrong order would be wrong behavior
for a real customer-service banking agent.

## Classification Rule

- `STRICT_ORDER`: the later write depends on state created by the earlier write,
  or the earlier write is a risk-control action that must exist before the later
  write is valid. Wrong order should fail in DB/replay semantics.
- `PREFERRED_ORDER`: the order is the better service workflow, but reversing it
  in the same episode does not clearly create an invalid bank-side operation.
- `COMMUTATIVE`: both writes are independent for policy, customer outcome, and
  DB state; order should not be evaluated.

## Current Scope

- Exported tasks: 123
- Tasks with at least two assistant-side write actions: 3
- `STRICT_ORDER`: 3
- `PREFERRED_ORDER`: 0
- `COMMUTATIVE`: 0

## Findings

| Task | Expected write order | Classification | Reality judgment | DB/replay enforcement |
| --- | --- | --- | --- | --- |
| `kb_manual_group_account_extra_pocket_auto_covers_withdrawal` | `execute_deposit_or_box_transfer` from additional pocket to basic pocket, then `execute_deposit_or_box_transfer` from basic pocket to external payee | `STRICT_ORDER` | The customer asks to cover a 150,000 KRW shortage before withdrawing 250,000 KRW. Withdrawing first from a basic pocket that only has 100,000 KRW is not a valid banking operation. | Enforced by balance-backed debit precondition. Reversed order now fails with insufficient balance. |
| `kb_manual_minicard_lost_card_compensation_within_60_days_success` | `update_card_state(REPORT_LOST_CARD)`, then `file_dispute_or_objection` for lost-card compensation | `STRICT_ORDER` | Lost-card compensation eligibility is anchored to the recorded report time. A compensation dispute that relies on a report timestamp before the card has been reported lost is not a valid workflow. It also delays the immediate risk-control action. | Enforced by lost-card compensation precondition. Reversed order now fails unless the card is already `LOST_REPORTED` with `lost_reported_at`. |
| `kb_manual_record_book_last_section_close_closes_account` | transfer section balance to linked account, then close the last non-interest section and record account | `STRICT_ORDER` | Once the final section is closed, transferring from that closed/zero-balance section as a separate later operation is invalid. If close is modeled as an atomic operation that also transfers the balance, the task could be represented with a single close write, but under the current two-write decomposition the explicit transfer must come first. | Enforced by balance-backed debit precondition. Reversed order now fails because the close zeroes the section balance first. |

## Validation

After enforcing DB/replay preconditions:

- Gold replay: `123/123` passed.
- Reversed write order for all 3 multi-write tasks: `3/3` failed.

Observed reversed-order failures:

- `kb_manual_group_account_extra_pocket_auto_covers_withdrawal`: insufficient balance, available `100000`, amount `250000`.
- `kb_manual_minicard_lost_card_compensation_within_60_days_success`: lost-card compensation dispute requires a recorded lost-card report first.
- `kb_manual_record_book_last_section_close_closes_account`: insufficient balance, available `0`, amount `180000`.

## Design Decision

For v0 DB-only evaluation, order-sensitive behavior should be represented in
DB/replay tool semantics whenever possible. That keeps the reward aligned with
the final DB-hash philosophy instead of introducing style matching over the
expected action list.

Action-order assertions should be reserved for cases where the order matters to
the real workflow but cannot be represented by durable state, failed tool
preconditions, or final DB differences.
