"""Hackathon v0 assistant runner for KakaoBank DB-delta tasks."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from kb_knowledge.kakaobank.data_model import (
    ACTION_VERIFIER_SCHEMA_PATH,
    KakaoBankDB,
    load_action_verifier_schema,
)
from kb_knowledge.kakaobank.replay import (
    DbEvaluationResult,
    _action_schema_by_name,
    apply_task_initial_state,
    build_empty_domain_db,
    evaluate_candidate_actions,
    replay_expected_action,
    replay_expected_actions,
)
from kb_knowledge.kakaobank.tools import (
    BM25_RETRIEVAL_CONFIGS,
    DEFAULT_RETRIEVAL_CONFIG,
    GREP_RETRIEVAL_CONFIGS,
    KakaoBankReadTools,
    SUPPORTED_RETRIEVAL_CONFIGS,
)

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DONE_TOOL_NAME = "done"
STOP_TOKEN = "###STOP###"
RUNNER_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a task request from the user.
Plan the task, search policy documents when needed, inspect runtime DB state, call the appropriate tools, and then stop.

Stop when you consider that you have solved the task.
To do so, send a message containing a single tool call to the `{done_tool_name}` tool. Do not include any other tool calls in this last message.

Always follow the policy. Always generate valid JSON tool arguments only.
""".strip()
RUNNER_SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<runtime_context>
{runtime_context}
</runtime_context>
""".strip()
KAKAOBANK_POLICY_HEADER = """
# KakaoBank Customer Service Policy

You are a helpful customer service agent for KakaoBank.
Your goal is to solve the customer's request by grounding the decision in KakaoBank policy documents and runtime DB state, then performing the allowed bank-side operation with tools.

## Guidelines

1. Do not make up policies, product rules, record IDs, operation codes, transaction types, close types, reasons, or actions that you can take on behalf of the user.
2. Use exact runtime record IDs from <runtime_context> or previous tool results. Do not invent IDs.
3. Read the relevant runtime DB records before write actions when balances, statuses, dates, linked products, consent state, restrictions, ownership, age, limits, or prior transactions matter.
4. If a requested operation is allowed by policy and the runtime state, perform the bank-side operation with the available write tools. Do not merely explain that it is possible, and do not ask whether to proceed, unless required information is truly missing.
5. If the request or policy requires multiple state changes, call all required write tools in the correct order before stopping.
6. Use concise, stable, code-like values for `operation`, `transaction_type`, `transfer_type`, `close_type`, and `reason`. Prefer the most specific value that matches the retrieved policy and runtime state.
""".strip()
KAKAOBANK_SINGLE_TURN_RUNNER_POLICY = """
## Single-Turn Runner Mode

This runner starts from the initial user prompt and does not provide additional user turns during the episode.
If the policy and runtime state contain enough information to complete an allowed operation, perform the operation with tools instead of asking for confirmation.

The user is treated as already authenticated for this runner mode. Do not perform identity verification unless a provided tool and task explicitly require it.
Only use tools exposed in this episode. Do not refer to unavailable user-side, discovery, unlock, or human-transfer tools as callable tools.

A task is complete only after all required read/write tool calls are done and you call `done` as the only tool call in the final assistant message.
""".strip()
KAKAOBANK_V0_EVALUATION_POLICY = """
## DB-Delta Evaluation Mode

This benchmark evaluates the final DB state after the complete assistant tool-call trajectory.
Intermediate DB hashes are not success signals. Continue until the requested operation is fully complete, then call `done`.

Every exported v0 task requires at least one assistant-side state-changing operation.
Plain assistant text is not a completion signal in this runner.
""".strip()

READ_TOOL_NAMES = {
    "KB_search",
    "grep",
    "get_customer_profile",
    "get_account_or_contract",
}
KAKAOBANK_TABLE_NAMES = (
    "customers",
    "businesses",
    "consents",
    "accounts",
    "deposit_contracts",
    "savings_boxes",
    "auto_transfer_rules",
    "group_memberships",
    "pockets",
    "child_relationships",
    "cards",
    "card_orders",
    "prepaid_wallets",
    "loans",
    "loan_applications",
    "refinance_requests",
    "required_documents",
    "mortgage_collateral",
    "lease_contracts",
    "vehicle_purchase_cases",
    "comparison_sessions",
    "remittance_profiles",
    "remittance_cases",
    "service_enrollments",
    "transactions",
    "disputes",
)
TOOL_DESCRIPTION_OVERRIDES = {
    "KB_search": (
        "Search the KakaoBank knowledge base with BM25 over document titles and "
        "content. Use this for broad natural-language product, policy, eligibility, "
        "limit, fee, deadline, exception, or procedure questions. Returns ranked "
        "documents with id, title, content, and score. Read-only; does not change DB state."
    ),
    "grep": (
        "Search the same KakaoBank knowledge documents with a regex or exact phrase. "
        "Use this for exact Korean terms, product names, quoted clauses, identifiers, "
        "amounts, dates, or when BM25 needs confirmation. Returns matching documents "
        "ranked by match count. Read-only; does not change DB state."
    ),
    "get_customer_profile": (
        "Read one customer's runtime profile, business records, and consent records. "
        "Use before state-changing actions when age, residency, identity status, "
        "business representative authority, guardian consent, affiliate consent, or "
        "terms consent could affect eligibility. Read-only."
    ),
    "get_account_or_contract": (
        "Read one runtime DB record by table and record_id. Use this to inspect the "
        "current state of accounts, deposits, boxes, loans, cards, wallets, services, "
        "remittance cases, documents, or transactions before deciding a write action. "
        "Read-only."
    ),
    "open_or_enroll_product": (
        "Create, enroll, convert, restrict, or reactivate a KakaoBank product or "
        "service after policy and runtime-state checks pass. Use for new demand or "
        "business accounts, SafeBox/Piggy/VAT/Dollar boxes, mini wallets, cards, "
        "credit-info services, group-account services, record-book accounts, and "
        "similar enrollment flows. Do not use for ordinary money movement, remittance "
        "settlement, loan status changes, or card state updates."
    ),
    "close_account_or_service": (
        "Close or terminate an existing account, deposit, box, wallet, service, or "
        "membership when closure rules allow it; also records supported policy-driven "
        "service stops or consent withdrawals. Check linked products, balances, legal "
        "restrictions, member state, age rules, and required destinations before calling."
    ),
    "configure_auto_transfer": (
        "Create, update, cancel, or reject an auto-transfer or scheduled saving rule. "
        "Use for 26-week savings, child savings, free savings, Piggy Bank auto-saving, "
        "and VAT/collection-style automatic transfers when the schedule or debit rule "
        "itself is the required state change."
    ),
    "execute_deposit_or_box_transfer": (
        "Execute or record an allowed deposit, withdrawal, refund, savings payment, "
        "box transfer, pocket movement, record-book section movement, or mini balance "
        "movement. Use this when money or balance-backed state should change. For "
        "rejected or pending movements, include the rejection/pending status in the "
        "arguments so replay leaves balances unchanged when appropriate."
    ),
    "request_interest_payment": (
        "Request and apply a product-specific interest payment, such as a SafeBox "
        "customer-requested interest payout. Use only when the product rule allows an "
        "interest-payment event distinct from ordinary monthly interest posting."
    ),
    "request_maturity_or_extension": (
        "Apply or reject deposit maturity close, holiday maturity close, early close, "
        "auto-close, auto-extension, auto-redeposit, or loan/deposit extension behavior. "
        "Use when maturity, renewal, payout destination, preferential-rate eligibility, "
        "or legal restriction state determines the outcome."
    ),
    "create_loan_application": (
        "Create a direct loan application or a loan-comparison partner handoff. Use "
        "when the customer is not yet changing an existing loan contract but a new "
        "application, quote, comparison session, or partner handoff record must exist."
    ),
    "update_loan_contract_state": (
        "Update an existing loan or loan-related case: execution, blocked execution, "
        "contract withdrawal, immediate repayment acceleration, collateral enforcement, "
        "document-deadline consequence, lease-loan disbursement, mortgage proof state, "
        "or used-car title-transfer verification. Use for high-risk post-contract loan "
        "state changes, not for creating a new application."
    ),
    "process_refinance_request": (
        "Process a refinance or loan-switching request and related old/new loan state. "
        "Use when repayment of the old loan, activation/cancellation of the new loan, "
        "lien release, same-day duplicate handling, or unrecoverable old-loan conditions "
        "determine the outcome."
    ),
    "execute_remittance_case": (
        "Execute, cancel, reject, return, hold, or document-request an inbound, outbound, "
        "or DollarBox gift remittance case. Use when remittance status, annual usage, "
        "fees, exchange-rate settlement, correction, return, or account/box balance "
        "must change."
    ),
    "update_card_state": (
        "Update mini-card/card state for issue, reissue, lost-card reporting, card "
        "restriction, rejected issuance, payment approval/refusal, or T-money transfer "
        "handling. Use when age, guardian consent, existing-card count, wallet balance, "
        "password errors, merchant restrictions, or report-window rules control the outcome."
    ),
    "file_dispute_or_objection": (
        "Create or update a dispute, objection, compensation, or affiliate-service "
        "complaint record. Use when the correct bank action is to log an investigation "
        "or dispute outcome rather than directly changing the underlying product state."
    ),
}
TOOL_ARGUMENT_OVERRIDES = {
    "get_account_or_contract": ("record_id", "table"),
    "close_account_or_service": (
        "customer_id",
        "target_id",
        "close_type",
        "reason",
        "destination_account_id",
        "also_close_service_id",
        "options",
    ),
    "execute_deposit_or_box_transfer": (
        "source_id",
        "source_account_id",
        "target_id",
        "amount",
        "currency",
        "transaction_type",
        "transfer_type",
        "week_number",
        "counts_as_auto_debit_success",
        "requested_by_customer_id",
        "representative_role",
        "emergency_withdrawal_count_after",
        "apply_rate",
        "remaining_principal_krw",
        "interest_rate_type_for_withdrawn_amount",
        "new_current_month_contribution_krw",
        "rejected_source_account_ids",
        "rejection_reason_for_other_sources",
        "payment_date",
        "new_cumulative_payment_count",
        "new_today_payment_count",
        "new_daily_preferential_rate_pp",
        "new_bonus_preferential_rate_pp",
        "counts_for_preferential_rate",
        "maturity_close_required_for_preferential",
        "new_transaction_status",
        "current_balance_krw",
        "holding_limit_krw",
        "would_be_balance_krw",
        "credited_to_wallet",
        "original_transfer_status",
        "new_transfer_status",
        "sent_at",
        "auto_cancel_at",
        "refunded_at",
        "previous_day_final_balance_krw",
        "saving_time_source_balance_krw",
        "options",
    ),
    "update_loan_contract_state": (
        "loan_id",
        "application_id",
        "operation",
        "reason",
        "effective_at",
        "options",
    ),
}
ARGUMENT_DESCRIPTION_OVERRIDES = {
    ("KB_search", "query"): (
        "Natural-language Korean or English search query describing the policy, product, "
        "eligibility rule, fee, limit, deadline, exception, or procedure to find. Do not "
        "include gold document IDs."
    ),
    ("grep", "pattern"): (
        "Regex or exact phrase to search in document titles and content. Use concise "
        "terms such as a product name, clause phrase, identifier, amount, date, or "
        "Korean keyword. Invalid regex is treated as a literal pattern."
    ),
    ("get_account_or_contract", "record_id"): (
        "Exact runtime record ID from the system context or a previous tool result, "
        "for example an account_id, deposit_id, box_id, loan_id, service_id, card_id, "
        "remittance_id, document_id, transaction_id, or case ID."
    ),
    ("get_account_or_contract", "table"): (
        "Name of the DB table that contains record_id. Prefer the table shown in the "
        "runtime context; use the enum values exactly."
    ),
    ("open_or_enroll_product", "options"): (
        "Object with operation-specific details such as operation/opening_mode, new "
        "record IDs, expected_status, consent IDs, limit/account/box/service fields, "
        "or rejection flags needed to make the replayed state match the policy outcome."
    ),
    ("close_account_or_service", "close_type"): (
        "Closure or termination code, for example NORMAL_CLOSE, CLOSE_SAFEBOX, "
        "CLOSE_PIGGY_BANK, CLOSE_VAT_BOX, CLOSE_DOLLARBOX, CLOSE_DEMAND_DEPOSIT_ACCOUNT, "
        "CLOSE_GROUP_ACCOUNT_SERVICE, REMOVE_GROUP_MEMBER, AUTO_CLOSE, FEE_CHANGE_CANCEL, "
        "AFFILIATE_SERVICE_STOP, WITHDRAW_SERVICE_CONSENT, or SERVICE_CONSENT_WITHDRAWAL."
    ),
    ("close_account_or_service", "options"): (
        "Object with closure-specific details such as destination account, balance or "
        "principal transfer amount, service/member IDs, notice/effective dates, release "
        "flags, pending refunds, or other fields required by the closure rule."
    ),
    ("close_account_or_service", "destination_account_id"): (
        "Account ID that receives principal, interest, balance, or remaining funds during "
        "a close action, when the closure policy requires a destination account."
    ),
    ("close_account_or_service", "also_close_service_id"): (
        "Related service ID to close together with the main target, when the policy links "
        "the account close and service close."
    ),
    ("configure_auto_transfer", "operation"): (
        "Auto-transfer operation code, such as CREATE, UPDATE, CANCEL, or REJECT, matching "
        "the requested schedule change and policy outcome."
    ),
    ("configure_auto_transfer", "schedule"): (
        "Structured schedule for the rule, including type, first_run_date, weekdays, "
        "requested/applied timestamp, and any holiday/deadline metadata relevant to the task."
    ),
    ("configure_auto_transfer", "options"): (
        "Object with auto_transfer_id, saving range, min/max amount, deadline, rejection, "
        "or expected_status fields needed for this scheduled transfer operation."
    ),
    ("execute_deposit_or_box_transfer", "options"): (
        "Object with transfer-specific replay details such as purchase/case IDs, final "
        "status, post-balance, pending/refund flags, or extra facts not represented by "
        "the top-level amount/source/target fields."
    ),
    ("execute_deposit_or_box_transfer", "source_account_id"): (
        "Account ID used as the money source. Use this instead of source_id when the "
        "task or runtime context names the source as an account."
    ),
    ("execute_deposit_or_box_transfer", "transfer_type"): (
        "Specific transfer subtype, for example MISSED_WEEK_FILL, when transaction_type "
        "is not the policy's most precise label."
    ),
    ("execute_deposit_or_box_transfer", "credited_to_wallet"): (
        "Whether the attempted transfer actually credits the wallet. False means replay "
        "should record a pending/refused movement without increasing wallet balance."
    ),
    ("execute_deposit_or_box_transfer", "counts_as_auto_debit_success"): (
        "Whether a savings payment counts as an automatic debit success for preferential "
        "rate or streak rules."
    ),
    ("execute_deposit_or_box_transfer", "counts_for_preferential_rate"): (
        "Whether this payment counts toward preferential-rate qualification."
    ),
    ("execute_deposit_or_box_transfer", "maturity_close_required_for_preferential"): (
        "Whether the preferential rate depends on a later maturity close."
    ),
    ("execute_deposit_or_box_transfer", "rejected_source_account_ids"): (
        "List or encoded value of rejected source account IDs when only some funding "
        "sources are allowed by policy."
    ),
    ("execute_deposit_or_box_transfer", "rejection_reason_for_other_sources"): (
        "Policy reason that non-allowed funding sources were rejected."
    ),
    ("update_loan_contract_state", "application_id"): (
        "Related loan application ID when the loan-state operation is tied to an "
        "application, execution, or withdrawal-right record."
    ),
    ("request_maturity_or_extension", "operation"): (
        "Maturity or extension operation code, such as MATURE_DIRECT_CLOSE, "
        "MATURE_HOLIDAY_PREVIOUS_BUSINESS_DAY_CLOSE, MATURE_AUTO_CLOSE, EARLY_CLOSE, "
        "AUTO_EXTEND, or AUTO_REDEPOSIT."
    ),
    ("request_interest_payment", "options"): (
        "Object with interest_amount_krw, destination_id, add_to_principal, reason, "
        "or other product-specific payout details."
    ),
    ("request_maturity_or_extension", "options"): (
        "Object with close_type, destination_account_id, payout amount, maturity/close "
        "dates, preferential-rate details, auto-extension count, legal restrictions, "
        "or rejection reason."
    ),
    ("update_loan_contract_state", "options"): (
        "Object with loan-specific details such as immediate_repayment_required, document "
        "status/deadline, landlord account, disbursement amount, collateral state, old loan "
        "repayment proof, title-transfer state, or execution/refusal metadata."
    ),
    ("update_loan_contract_state", "operation"): (
        "Loan-state operation code, such as ACCELERATE_IMMEDIATE_REPAYMENT, "
        "MARK_EXECUTION_BLOCKED, STOP_EXECUTION_BEFORE_DISBURSEMENT, EXECUTE_LEASE_LOAN_TO_LANDLORD, "
        "RESUME_EXECUTION_WITH_UPDATED_LANDLORD_ACCOUNT, PROCESS_LOAN_WITHDRAWAL_RIGHT, "
        "VERIFY_USED_CAR_TITLE_TRANSFER, START_COLLATERAL_ENFORCEMENT, or another precise "
        "policy-grounded loan-state transition."
    ),
    ("process_refinance_request", "operation"): (
        "Refinance operation code, such as COMPLETE_OLD_LOAN_REPAYMENT_AND_ACTIVATE_NEW_LOAN, "
        "CANCEL_NEW_LOAN_AFTER_OLD_REPAYMENT_FAILURE, REJECT_UNREPAYABLE_OLD_LOAN, or "
        "COMPLETE_OLD_LOAN_REPAYMENT_AND_REQUEST_LIEN_RELEASE."
    ),
    ("process_refinance_request", "options"): (
        "Object with refinance-specific details such as old/new loan status, reason, "
        "home type, lien release status, cancellation state, or repayment verification."
    ),
    ("execute_remittance_case", "direction"): (
        "Remittance case direction or flow code, such as INBOUND_IMMEDIATE_DEPOSIT, "
        "INBOUND_BULK_DEPOSIT, INBOUND_RETURN_INFO_MISMATCH, OUTBOUND_NO_DOCUMENT, "
        "OUTBOUND_BUSINESS_PURPOSE_REJECTED, OUTBOUND_RETURN_SETTLEMENT, "
        "DOLLARBOX_GIFT_RECEIVE, or DOLLARBOX_GIFT_AUTO_CANCEL."
    ),
    ("execute_remittance_case", "options"): (
        "Object with remittance_id and case-specific fields such as source/target account, "
        "box IDs, transaction IDs, fees, exchange rates, document-request reason, expected "
        "status, return/cancel reason, or refund/deposit metadata."
    ),
    ("update_card_state", "operation"): (
        "Card operation code, such as ISSUE_NEW_CARD, REISSUE_CARD, REPORT_LOST_CARD, "
        "REJECT_NEW_ISSUE, RESTRICT_CARD_AND_REJECT_TRANSACTION, or "
        "REISSUE_CARD_WITHOUT_TMONEY_TRANSFER."
    ),
    ("update_card_state", "options"): (
        "Object for rare card-state fields not represented by top-level arguments. Prefer "
        "top-level card_id, operation, reason, order_id, wallet_id, amount, and status fields "
        "when available."
    ),
    ("file_dispute_or_objection", "options"): (
        "Object with dispute-specific outcome fields such as compensation_approved, "
        "reactivation_approved, service_access_allowed, investigation_status, fault flags, "
        "liability scope, or affiliate-resolution metadata."
    ),
}
ARGUMENT_DESCRIPTIONS = {
    "customer_id": "Exact customer_id from the runtime context or customer profile result.",
    "record_id": "Exact ID of the record to read or modify.",
    "table": "Runtime DB table name containing the target record.",
    "source_account_id": "Account ID used as the funding, base, linked, or originating account.",
    "source_id": "ID of the source record to debit or move value from, when applicable.",
    "target_id": "ID of the account, contract, box, loan, service, wallet, card, case, or document affected by the action.",
    "target_type": "Type of target being disputed or updated, such as service_enrollment, card, wallet, transaction, loan, or remittance_case.",
    "product_name": "KakaoBank product or service name exactly as used in the task or retrieved policy.",
    "close_type": "Specific closure, termination, consent-withdrawal, removal, or rejection code that explains the closure path.",
    "reason": "Short reason code or policy-grounded explanation for this action. Use a precise stable code, not a long paragraph.",
    "schedule": "Structured schedule object for an auto-transfer or recurring saving rule.",
    "amount": "Money amount for the transaction, remittance, card payment, or balance movement.",
    "amount_krw": "KRW amount for an auto-transfer, payment, fee, limit, or balance movement. Use null only when the rule has a variable amount.",
    "currency": "Currency code such as KRW or USD.",
    "transaction_type": "Specific transaction or movement type code, such as savings payment, emergency withdrawal, box transfer, refund, or auto-collect.",
    "transfer_type": "Specific transfer subtype when transaction_type is not used.",
    "payment_date": "Date on which a payment or contribution is applied.",
    "requested_at": "ISO-like timestamp for when the customer or system requested the operation.",
    "effective_at": "ISO-like timestamp when the state change, restriction, acceleration, or status update takes effect.",
    "operation": "Specific operation code to apply, reject, close, extend, accelerate, issue, reissue, activate, cancel, or verify the target state.",
    "options": "Operation-specific structured details needed for deterministic replay.",
    "application_id": "ID for the loan application record to create or update.",
    "comparison_id": "ID for a loan-comparison session or partner handoff, when applicable.",
    "requested_amount_krw": "Requested loan amount in KRW.",
    "purpose": "Customer's declared product, loan, remittance, or transfer purpose code.",
    "partner_id": "Partner institution or comparison-service partner ID, or null if no partner is involved.",
    "expected_status": "Expected resulting status for the new or updated record.",
    "expected_reason": "Expected status reason or policy reason for the application/handoff result.",
    "loan_id": "ID of the existing loan or loan contract affected by the operation.",
    "refinance_id": "ID of the refinance or loan-switching request.",
    "old_loan_repayment_status": "Repayment status of the old loan in a refinance flow.",
    "direction": "Remittance flow or case direction, such as inbound, outbound, return, hold, or DollarBox gift flow.",
    "country": "Country code or destination/source country for the remittance case.",
    "purpose_code": "Remittance purpose code, such as living expense, personal transfer, ad revenue, gift, or business purpose.",
    "card_id": "ID of the card affected by the operation, or null for a rejected new issue before a card exists.",
    "wallet_id": "ID of the prepaid mini wallet or related wallet record.",
    "order_id": "ID of the card order or issuance/reissuance request.",
    "existing_card_id": "ID of an existing card relevant to issue/reissue/card-count checks.",
    "new_card_created": "Whether the operation creates a new card record.",
    "expected_valid_until": "Expected card validity end date after issue or reissue.",
    "new_status": "Expected resulting status of the affected existing record.",
    "new_card_status": "Expected status of a newly created or reissued card.",
    "new_card_id": "ID of a newly created card record.",
    "attempted_transaction_type": "Type of attempted card or wallet transaction being approved or rejected.",
    "transaction_approved": "Whether the attempted transaction is approved.",
    "payment_approved": "Whether the attempted card payment is approved.",
    "amount_currency": "Currency code for the amount field when both KRW and foreign-currency amounts are present.",
    "amount_original": "Original non-KRW or source-currency amount for a card/payment/remittance case.",
    "merchant_country": "Merchant country code for card payment eligibility checks.",
    "merchant_currency": "Merchant currency code for card payment eligibility checks.",
    "merchant_category": "Merchant category or industry code/name for restricted-card-payment checks.",
    "merchant_category_basis": "Policy basis for treating the merchant category as allowed or restricted.",
    "old_t_money_balance_krw": "Previous T-money balance in KRW before card reissue handling.",
    "new_card_t_money_balance_krw": "T-money balance assigned to the new card after reissue handling.",
    "t_money_balance_transferred": "Whether T-money balance is transferred to the new card.",
    "t_money_refund_processed": "Whether T-money refund processing occurred.",
    "customer_age": "Customer age used for age-gated card or mini-wallet checks.",
    "guardian_consent_valid": "Whether legal guardian consent is valid for the requested minor action.",
    "issue_fee_krw": "Card issue or reissue fee in KRW.",
}
NUMBER_ARGUMENTS = {
    "amount",
    "amount_krw",
    "requested_amount_krw",
    "issue_fee_krw",
    "customer_age",
    "old_t_money_balance_krw",
    "new_card_t_money_balance_krw",
}
BOOLEAN_ARGUMENTS = {
    "new_card_created",
    "transaction_approved",
    "payment_approved",
    "t_money_balance_transferred",
    "t_money_refund_processed",
    "guardian_consent_valid",
    "counts_as_auto_debit_success",
    "counts_for_preferential_rate",
    "maturity_close_required_for_preferential",
    "credited_to_wallet",
}
ARRAY_ARGUMENTS = {
    "rejected_source_account_ids",
}


class ChatClient(Protocol):
    """Small protocol for OpenAI-compatible chat-completions clients."""

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
    ) -> dict[str, Any]:
        """Return a chat-completions response dict."""


@dataclass
class AssistantRunResult:
    """End-to-end assistant run and DB evaluation result."""

    task_id: str
    passed: bool
    actions: list[dict[str, Any]]
    final_text: str
    evaluation: DbEvaluationResult
    stopped_reason: str
    expected_final_hash: str
    actual_final_hash: str | None
    error: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleChatClient:
    """Minimal requests-based OpenAI-compatible Chat Completions client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
        base_url: str | None = None,
        timeout_seconds: int = 90,
    ) -> None:
        self.api_key = api_key or os.environ.get(api_key_env)
        self.chat_completions_url = _chat_completions_url(base_url or endpoint)
        self.timeout_seconds = timeout_seconds

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.chat_completions_url,
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": temperature,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


OpenAIChatClient = OpenAICompatibleChatClient


def run_task_with_openai_compatible(
    task_data: dict[str, Any],
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    endpoint: str = DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    retrieval_config: str = DEFAULT_RETRIEVAL_CONFIG,
    max_tool_steps: int = 12,
    timeout_seconds: int = 90,
) -> AssistantRunResult:
    """Run one task through an OpenAI-compatible chat server and evaluate DB."""

    return run_task_with_chat_client(
        task_data,
        chat_client=OpenAICompatibleChatClient(
            endpoint=endpoint,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        ),
        model=model,
        retrieval_config=retrieval_config,
        max_tool_steps=max_tool_steps,
    )


run_task_with_openai = run_task_with_openai_compatible


def run_task_with_chat_client(
    task_data: dict[str, Any],
    *,
    chat_client: ChatClient,
    model: str = DEFAULT_OPENAI_MODEL,
    retrieval_config: str = DEFAULT_RETRIEVAL_CONFIG,
    max_tool_steps: int = 12,
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
) -> AssistantRunResult:
    """Run one task with an injected chat client for testable tool chaining."""

    if retrieval_config not in SUPPORTED_RETRIEVAL_CONFIGS:
        raise ValueError(
            f"unknown retrieval_config: {retrieval_config!r}; "
            f"supported: {', '.join(SUPPORTED_RETRIEVAL_CONFIGS)}"
        )
    expected = replay_expected_actions(task_data, schema_path=schema_path)
    db = apply_task_initial_state(build_empty_domain_db(), task_data)
    read_tools = KakaoBankReadTools(db, retrieval_config=retrieval_config)
    schema_actions = _action_schema_by_name(schema_path)
    tools = build_openai_tool_definitions(
        schema_path=schema_path,
        retrieval_config=retrieval_config,
    )
    tool_names = [str(tool["function"]["name"]) for tool in tools]
    messages = [
        {
            "role": "system",
            "content": build_runner_system_prompt(
                task_data,
                retrieval_config=retrieval_config,
            ),
        },
        {
            "role": "user",
            "content": str(task_data["user_prompt"]),
        },
    ]

    actions: list[dict[str, Any]] = []
    final_text = ""
    stopped_reason = "max_tool_steps"
    error: str | None = None
    trace: dict[str, Any] = {
        "schema_version": "kakaobank_assistant_trace.v0",
        "task_id": str(task_data["id"]),
        "model": model,
        "retrieval_config": retrieval_config,
        "max_tool_steps": max_tool_steps,
        "initial_db_hash": db.get_hash(),
        "expected_final_hash": expected.final_hash,
        "system_prompt": messages[0]["content"],
        "user_prompt": messages[1]["content"],
        "available_tools": tools,
        "tool_names": tool_names,
        "rounds": [],
    }

    for round_index in range(max_tool_steps + 1):
        round_trace: dict[str, Any] = {
            "round_index": round_index,
            "request": {
                "message_count": len(messages),
                "messages": copy.deepcopy(messages),
                "tool_names": tool_names,
                "temperature": 0.0,
            },
            "response": {},
            "tool_results": [],
        }
        started_at = time.perf_counter()
        response = chat_client.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.0,
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        assistant_message = _extract_assistant_message(response)
        choice = (response.get("choices") or [{}])[0]
        tool_calls = assistant_message.get("tool_calls") or []
        final_text = str(assistant_message.get("content") or "")
        round_trace["response"] = {
            "assistant_message": assistant_message,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "duration_ms": duration_ms,
        }
        trace["rounds"].append(round_trace)

        if _has_done_tool_call(tool_calls):
            stopped_reason = "agent_stop"
            final_text = STOP_TOKEN
            round_trace["stop_signal"] = _done_tool_call_trace(tool_calls)
            break

        if not tool_calls:
            stopped_reason = "final_answer"
            break

        messages.append(_assistant_message_for_history(assistant_message))
        for tool_call in tool_calls:
            action = _action_from_tool_call(tool_call)
            actions.append(action)
            tool_result = execute_runner_tool(
                db,
                action,
                read_tools=read_tools,
                schema_actions=schema_actions,
                task_id=str(task_data["id"]),
                action_index=len(actions) - 1,
            )
            if "error" in tool_result and error is None:
                error = str(tool_result["error"])
            round_trace["tool_results"].append(
                {
                    "action_index": len(actions) - 1,
                    "tool_call_id": str(tool_call.get("id", f"tool_{len(actions)}")),
                    "name": action["name"],
                    "arguments": action["arguments"],
                    "result": tool_result,
                    "db_hash_after": db.get_hash(),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id", f"tool_{len(actions)}")),
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

    evaluation = evaluate_candidate_actions(
        task_data,
        actions,
        schema_path=schema_path,
    )
    passed = evaluation.passed and stopped_reason == "agent_stop"
    termination_error = None
    if stopped_reason != "agent_stop":
        termination_error = (
            f"episode did not terminate with agent_stop: {stopped_reason}"
        )
    trace["final"] = {
        "passed": passed,
        "db_passed": evaluation.passed,
        "stopped_reason": stopped_reason,
        "action_count": len(actions),
        "final_text": final_text,
        "expected_final_hash": evaluation.expected_final_hash,
        "actual_final_hash": evaluation.actual_final_hash,
        "error": error or evaluation.error or termination_error,
    }
    return AssistantRunResult(
        task_id=str(task_data["id"]),
        passed=passed,
        actions=actions,
        final_text=final_text,
        evaluation=evaluation,
        stopped_reason=stopped_reason,
        expected_final_hash=evaluation.expected_final_hash,
        actual_final_hash=evaluation.actual_final_hash,
        error=error or evaluation.error or termination_error,
        trace=trace,
    )


def execute_runner_tool(
    db: KakaoBankDB,
    action: dict[str, Any],
    *,
    read_tools: KakaoBankReadTools,
    schema_actions: dict[str, dict[str, Any]],
    task_id: str,
    action_index: int,
) -> dict[str, Any]:
    """Execute one assistant tool call against the runner DB."""

    name = str(action.get("name", ""))
    arguments = action.get("arguments") or {}
    if not isinstance(arguments, dict):
        return {"error": f"tool arguments must be an object for {name}"}

    try:
        if name == "KB_search":
            return read_tools.KB_search(str(arguments.get("query", "")))
        if name == "grep":
            return read_tools.grep(
                str(arguments.get("pattern") or arguments.get("query") or "")
            )
        if name == "get_customer_profile":
            return read_tools.get_customer_profile(
                str(arguments.get("customer_id", ""))
            )
        if name == "get_account_or_contract":
            record_id = str(
                arguments.get("record_id")
                or arguments.get("target_id")
                or arguments.get("account_id")
                or ""
            )
            table = arguments.get("table")
            if not table:
                table = _find_table_for_record_id(db, record_id)
            return read_tools.get_account_or_contract(record_id, str(table))

        replayed = replay_expected_action(
            db,
            action,
            schema_actions=schema_actions,
            task_id=task_id,
            action_index=action_index,
        )
        return {
            "status": replayed.status,
            "mutates_state": replayed.mutates_state,
            "db_hash": db.get_hash(),
        }
    except Exception as exc:  # noqa: BLE001 - tool errors are returned to the model.
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "db_hash": db.get_hash(),
        }


def build_runner_system_prompt(
    task_data: dict[str, Any],
    *,
    retrieval_config: str = DEFAULT_RETRIEVAL_CONFIG,
) -> str:
    """Build the tau3-style system prompt used by the v0 runner."""

    agent_instruction = RUNNER_AGENT_INSTRUCTION.format(done_tool_name=DONE_TOOL_NAME)
    return RUNNER_SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=agent_instruction,
        domain_policy=build_runner_domain_policy(retrieval_config=retrieval_config),
        runtime_context=build_runtime_context(task_data),
    )


def build_runner_domain_policy(
    *,
    retrieval_config: str = DEFAULT_RETRIEVAL_CONFIG,
) -> str:
    """Build the reusable KakaoBank v0 domain policy section."""

    return "\n\n".join(
        [
            KAKAOBANK_POLICY_HEADER,
            _retrieval_policy_for_config(retrieval_config),
            KAKAOBANK_SINGLE_TURN_RUNNER_POLICY,
            KAKAOBANK_V0_EVALUATION_POLICY,
        ]
    )


def _retrieval_policy_for_config(retrieval_config: str) -> str:
    if retrieval_config == "bm25":
        access = (
            "Search the knowledge base for relevant information when appropriate "
            "using the provided `KB_search` tool. `KB_search` uses BM25 over "
            "document titles and content."
        )
    elif retrieval_config == "grep":
        access = (
            "Search the knowledge base for relevant information when appropriate "
            "using the provided `grep` tool. `grep` searches document titles and "
            "content with regex or exact phrases."
        )
    else:
        access = (
            "Search the knowledge base for relevant information when appropriate "
            "using the provided `KB_search` and `grep` tools. `KB_search` uses "
            "BM25 over document titles and content. Use `grep` for exact Korean "
            "terms, product names, operation clues, dates, amounts, exception "
            "phrases, and to confirm BM25 results."
        )

    return "\n".join(
        [
            "## Knowledge Base Access",
            "",
            access,
            "",
            "Search before writing when the correct operation depends on product policy, eligibility, exceptions, deadlines, linked-product behavior, compensation rules, close/maturity behavior, or multi-step state changes.",
            "Use retrieved policy facts to choose precise write tools and code-like arguments.",
        ]
    )


def build_runtime_context(task_data: dict[str, Any]) -> str:
    """Return visible runtime IDs from task initialization data."""

    initialization_data = (task_data.get("initial_state") or {}).get(
        "initialization_data"
    ) or {}
    agent_data = initialization_data.get("agent_data") or {}
    lines = [f"Task ID: {task_data.get('id', '')}", "Runtime DB record IDs:"]
    if not agent_data:
        lines.append("- none")
        return "\n".join(lines)

    for table_name in sorted(agent_data):
        table = agent_data[table_name]
        records = table.get("data") if isinstance(table, dict) else None
        if not isinstance(records, dict) or not records:
            continue
        record_ids = ", ".join(sorted(str(record_id) for record_id in records))
        lines.append(f"- {table_name}: {record_ids}")
    return "\n".join(lines)


def build_openai_tool_definitions(
    *,
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
    retrieval_config: str = DEFAULT_RETRIEVAL_CONFIG,
) -> list[dict[str, Any]]:
    """Build Chat Completions tool definitions for assistant-facing v0 tools."""

    if retrieval_config not in SUPPORTED_RETRIEVAL_CONFIGS:
        raise ValueError(
            f"unknown retrieval_config: {retrieval_config!r}; "
            f"supported: {', '.join(SUPPORTED_RETRIEVAL_CONFIGS)}"
        )
    schema = load_action_verifier_schema(schema_path)
    definitions: list[dict[str, Any]] = []
    for action_schema in schema["action_families"]:
        name = str(action_schema["name"])
        if action_schema.get("requestor") != "assistant":
            continue
        if name == "KB_search" and retrieval_config not in BM25_RETRIEVAL_CONFIGS:
            continue
        if name == "grep" and retrieval_config not in GREP_RETRIEVAL_CONFIGS:
            continue
        definitions.append(_openai_tool_definition(action_schema))
    definitions.append(_done_tool_definition())
    return definitions


def _done_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": DONE_TOOL_NAME,
            "description": (
                "Stop the episode after all required KakaoBank operations are complete. "
                "Call this exactly once as the only tool call in the assistant message "
                "when no more read or write tools are needed. This tool is an AGENT_STOP "
                "signal and does not change DB state."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def _openai_tool_definition(action_schema: dict[str, Any]) -> dict[str, Any]:
    name = str(action_schema["name"])
    argument_names = _runner_argument_names(action_schema)
    properties = {
        argument_name: _argument_json_schema(argument_name, tool_name=name)
        for argument_name in argument_names
    }
    required = list(argument_names) if name in READ_TOOL_NAMES else []

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _tool_description(action_schema),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": True,
            },
        },
    }


def _runner_argument_names(action_schema: dict[str, Any]) -> tuple[str, ...]:
    name = str(action_schema["name"])
    if name in TOOL_ARGUMENT_OVERRIDES:
        return TOOL_ARGUMENT_OVERRIDES[name]
    return tuple(str(argument) for argument in action_schema.get("arguments", []))


def _tool_description(action_schema: dict[str, Any]) -> str:
    name = str(action_schema["name"])
    if name in TOOL_DESCRIPTION_OVERRIDES:
        return TOOL_DESCRIPTION_OVERRIDES[name]

    description_parts = [str(action_schema.get("notes") or "")]
    postconditions = action_schema.get("postconditions") or []
    if postconditions:
        description_parts.append(" ".join(str(item) for item in postconditions))
    return " ".join(part for part in description_parts if part)


def _argument_json_schema(
    argument_name: str,
    *,
    tool_name: str,
) -> dict[str, Any]:
    description = _argument_description(argument_name, tool_name=tool_name)
    if argument_name in NUMBER_ARGUMENTS or argument_name.endswith("_krw"):
        return {"type": "number", "description": description}
    if argument_name in {"options", "schedule"}:
        return {
            "type": "object",
            "description": description,
            "additionalProperties": True,
        }
    if argument_name in ARRAY_ARGUMENTS:
        return {
            "type": "array",
            "description": description,
            "items": {"type": "string"},
        }
    if (
        argument_name in BOOLEAN_ARGUMENTS
        or argument_name.endswith("_valid")
        or argument_name.endswith("_approved")
    ):
        return {"type": "boolean", "description": description}
    if tool_name == "get_account_or_contract" and argument_name == "table":
        return {
            "type": "string",
            "description": description,
            "enum": list(KAKAOBANK_TABLE_NAMES),
        }
    return {"type": "string", "description": description}


def _argument_description(argument_name: str, *, tool_name: str) -> str:
    return (
        ARGUMENT_DESCRIPTION_OVERRIDES.get((tool_name, argument_name))
        or ARGUMENT_DESCRIPTIONS.get(argument_name)
        or f"Value for {argument_name}; use the exact runtime value required by the task and policy."
    )


def _has_done_tool_call(tool_calls: list[dict[str, Any]]) -> bool:
    return any(
        str((tool_call.get("function") or {}).get("name", "")) == DONE_TOOL_NAME
        for tool_call in tool_calls
    )


def _done_tool_call_trace(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    done_calls = [
        tool_call
        for tool_call in tool_calls
        if str((tool_call.get("function") or {}).get("name", "")) == DONE_TOOL_NAME
    ]
    return {
        "token": STOP_TOKEN,
        "tool_call": done_calls[0] if done_calls else None,
        "valid_single_tool_call": len(tool_calls) == 1 and len(done_calls) == 1,
    }


def _extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("chat response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat response choice has no assistant message")
    return message


def _assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    history_message = {
        "role": "assistant",
        "content": message.get("content"),
    }
    if message.get("tool_calls"):
        history_message["tool_calls"] = message["tool_calls"]
    return history_message


def _action_from_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    name = str(function.get("name", ""))
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments) if raw_arguments else {}
    else:
        arguments = raw_arguments
    if not isinstance(arguments, dict):
        raise ValueError(f"tool call arguments must be an object for {name}")
    return {
        "requestor": "assistant",
        "name": name,
        "arguments": arguments,
    }


def _find_table_for_record_id(db: KakaoBankDB, record_id: str) -> str:
    for table_name, table in db.iter_tables():
        if record_id in table.data:
            return table_name
    raise ValueError(f"record ID not found in initialized DB: {record_id}")


def _chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"
