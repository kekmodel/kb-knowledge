"""Pydantic argument models for assistant-facing KakaoBank tools."""

from __future__ import annotations

import copy
from typing import Any, Literal

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

JsonNumber = int | float


def _numeric_suffix(record_id: str) -> str:
    suffix = record_id.rsplit("_", 1)[-1]
    if len(suffix) == 3 and suffix.isdigit():
        return suffix
    raise ValueError(f"{record_id!r} does not end with a three-digit suffix")


def _validate_generated_id(
    *,
    source_id: str,
    field_name: str,
    actual: str | None,
    prefix: str,
) -> None:
    if actual is None:
        return
    expected = f"{prefix}_{_numeric_suffix(source_id)}"
    if actual != expected:
        raise ValueError(f"{field_name} must be {expected!r} for {source_id!r}")


class ToolArgumentModel(BaseModel):
    """Base class for strict tool argument validation."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RemittanceCommonArguments(ToolArgumentModel):
    customer_id: str = Field(description="Customer ID that owns the remittance case.")
    amount: JsonNumber = Field(description="Foreign-currency remittance amount.")
    currency: str = Field(description="Foreign-currency code, such as USD, EUR, JPY, or CAD.")
    country: str = Field(description="Source or destination country code for the remittance case.")
    purpose_code: Literal[
        "AD_REVENUE",
        "BUSINESS_VENDOR_PAYMENT",
        "GIFT_AUTO_CANCEL_AFTER_30_DAYS",
        "GIFT_RECEIVE_WITHIN_30_DAYS",
        "LIVING_EXPENSE",
        "PERSONAL_TRANSFER",
    ] = Field(
        description=(
            "DB mutation purpose code. Use the runtime purpose when visible. "
            "DollarBox gift auto-cancel uses GIFT_AUTO_CANCEL_AFTER_30_DAYS; "
            "DollarBox gift receive uses GIFT_RECEIVE_WITHIN_30_DAYS."
        )
    )


class DollarBoxGiftAutoCancelOptions(ToolArgumentModel):
    """Auto-cancel a DollarBox gift not received within 30 days and refund sender."""

    remittance_id: str
    sender_box_id: str
    recipient_box_id: str
    cancel_reason: Literal["RECIPIENT_NOT_RECEIVED_WITHIN_30_DAYS"]
    refund_transaction_id: str | None = Field(
        description=(
            "New refund transaction ID. Set null to let the tool generate it. "
            "If supplied manually, for remittance_id remit_gift_NNN use "
            "txn_dollar_gift_refund_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="refund_transaction_id",
            actual=self.refund_transaction_id,
            prefix="txn_dollar_gift_refund",
        )
        return self


class DollarBoxGiftReceiveOptions(ToolArgumentModel):
    """Complete a DollarBox gift receive after recipient real-name confirmation."""

    remittance_id: str
    sender_box_id: str
    recipient_box_id: str
    recipient_real_name_confirmed: bool
    receive_completed_at: str
    receive_transaction_id: str | None = Field(
        description=(
            "New receive transaction ID. Set null to let the tool generate it. "
            "If supplied manually, for remittance_id remit_gift_NNN use "
            "txn_dollar_gift_receive_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="receive_transaction_id",
            actual=self.receive_transaction_id,
            prefix="txn_dollar_gift_receive",
        )
        return self


class InboundAutoReceiveDocumentRequestOptions(ToolArgumentModel):
    """Request evidence before deposit when matched auto-receive still needs review."""

    remittance_id: str
    target_account_id: str
    auto_receive_matched: bool
    document_request_reason: Literal[
        "BUSINESS_ACCOUNT_REQUIRES_PURPOSE_CONFIRMATION_AND_EVIDENCE_BEFORE_DEPOSIT"
    ]
    expected_status: Literal["PENDING_DOCUMENT_REVIEW"]
    deposit_first_refused: bool
    transaction_id: None


class InboundDailyOver100kDocumentRequestOptions(ToolArgumentModel):
    """Request documents before deposit when daily inbound aggregate exceeds USD 100,000."""

    remittance_id: str
    target_account_id: str
    daily_received_usd_before_case: JsonNumber
    daily_aggregate_usd_after_case: JsonNumber
    document_request_reason: Literal["DAILY_AGGREGATE_OVER_100K_USD"]
    expected_status: Literal["PENDING_DOCUMENT_REVIEW"]
    deposit_first_refused: bool
    transaction_id: None


class InboundBulkDepositOptions(ToolArgumentModel):
    """Bulk-deposit an inbound remittance on the fourth business day without application."""

    remittance_id: str
    target_account_id: str
    bulk_deposit_reason: Literal["NO_RECEIVE_APPLICATION_4TH_BUSINESS_DAY"]
    deposit_date: str
    exchange_rate_krw_per_unit: JsonNumber
    credit_amount_krw: JsonNumber
    receive_fee_krw: JsonNumber
    fee_waiver_reason: Literal["PROMO_2024_10_01_TO_2026_09_30"]
    transaction_id: str | None = Field(
        description=(
            "New inbound deposit transaction ID. Set null to let the tool generate it. "
            "If supplied manually, for remittance_id remit_inbound_NNN use "
            "txn_inbound_remit_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="transaction_id",
            actual=self.transaction_id,
            prefix="txn_inbound_remit",
        )
        return self


class InboundReturnInfoMismatchOptions(ToolArgumentModel):
    """Return an inbound remittance because recipient or remittance information mismatches."""

    remittance_id: str
    target_account_id: str
    mismatch_review_result: str
    expected_status: Literal["RETURNED_INFO_MISMATCH"]
    deposit_first_refused: bool
    return_transaction_id: str | None = Field(
        description=(
            "New inbound return transaction ID. Set null to let the tool generate it. "
            "If supplied manually, for remittance_id remit_inbound_NNN use "
            "txn_inbound_return_NNN."
        )
    )
    deposit_transaction_id: None

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="return_transaction_id",
            actual=self.return_transaction_id,
            prefix="txn_inbound_return",
        )
        return self


class InboundResidencyVerificationHoldOptions(ToolArgumentModel):
    """Hold an inbound remittance because customer residency is not verified."""

    remittance_id: str
    target_account_id: str
    resident_verified: bool
    expected_status: Literal["PENDING_RESIDENCY_VERIFICATION"]
    hold_reason: Literal["RESIDENCY_NOT_VERIFIED_DELAY_OR_RETURN_POSSIBLE"]
    deposit_first_refused: bool
    transaction_id: None


class InboundImmediateDepositOptions(ToolArgumentModel):
    """Immediately deposit an inbound remittance allowed without receive application."""

    remittance_id: str
    target_account_id: str
    deposit_reason: Literal["UNDER_5000_USD_NO_RECEIVE_APPLICATION_REQUIRED"]
    deposit_date: str
    exchange_rate_krw_per_unit: JsonNumber
    credit_amount_krw: JsonNumber
    receive_fee_krw: JsonNumber
    fee_waiver_reason: Literal["PROMO_2024_10_01_TO_2026_09_30"]
    transaction_id: str | None = Field(
        description=(
            "New inbound deposit transaction ID. Set null to let the tool generate it. "
            "If supplied manually, for remittance_id remit_inbound_NNN use "
            "txn_inbound_remit_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="transaction_id",
            actual=self.transaction_id,
            prefix="txn_inbound_remit",
        )
        return self


class OutboundBeneficiaryInfoAutoCancelOptions(ToolArgumentModel):
    """Auto-cancel an outbound remittance after beneficiary correction deadline expires."""

    remittance_id: str
    source_account_id: str
    correction_requested_at: str
    correction_due_date: str
    processed_at: str
    business_days_elapsed: JsonNumber
    expected_status: Literal["AUTO_CANCELED"]
    return_exchange_rate_krw_per_unit: JsonNumber
    returned_principal_krw: JsonNumber
    send_fee_krw: JsonNumber
    send_fee_refunded: bool
    transaction_id: str | None = Field(
        description=(
            "New outbound auto-cancel return transaction ID. Set null to let the "
            "tool generate it. If supplied manually, for remittance_id "
            "remit_outbound_NNN use txn_outbound_auto_cancel_return_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="transaction_id",
            actual=self.transaction_id,
            prefix="txn_outbound_auto_cancel_return",
        )
        return self


class OutboundBusinessPurposeRejectedOptions(ToolArgumentModel):
    """Reject an outbound remittance that uses a business sender name or purpose."""

    remittance_id: str
    source_account_id: str
    requested_sender_name: str
    business_account_as_source: bool
    rejection_reason: Literal["BUSINESS_NAME_OR_BUSINESS_PURPOSE_REMITTANCE_NOT_ALLOWED"]
    expected_status: Literal["REJECTED"]
    transaction_id: None


class OutboundNoDocumentOptions(ToolArgumentModel):
    """Send an outbound no-document remittance after fees, limits, and FX are confirmed."""

    remittance_id: str
    source_account_id: str
    recipient_name: str
    recipient_country: str
    recipient_relationship: str
    applied_exchange_rate_krw_per_unit: JsonNumber
    fx_preference_rate_percent: JsonNumber
    remittance_amount_krw: JsonNumber
    send_fee_krw: JsonNumber
    total_debit_krw: JsonNumber
    wire_fee_waived: bool
    intermediary_and_recipient_fee_borne_by: str
    annual_usd_equivalent: JsonNumber
    new_annual_usd_sent: JsonNumber
    transaction_id: str | None = Field(
        description=(
            "New outbound remittance transaction ID. Set null to let the tool "
            "generate it. If supplied manually, for remittance_id "
            "remit_outbound_NNN use txn_outbound_remit_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="transaction_id",
            actual=self.transaction_id,
            prefix="txn_outbound_remit",
        )
        return self


class OutboundNoDocumentOver100kSingleLimitRejectedOptions(ToolArgumentModel):
    """Reject a no-document outbound remittance above the post-USD-100,000 case limit."""

    remittance_id: str
    source_account_id: str
    annual_usd_sent_before_case: JsonNumber
    requested_no_document_amount_usd: JsonNumber
    allowed_single_case_limit_usd_after_100k: JsonNumber
    rejection_reason: Literal["NO_DOCUMENT_AFTER_100K_SINGLE_CASE_OVER_5000_USD"]
    expected_status: Literal["REJECTED"]
    transaction_id: None


class OutboundReturnSettlementOptions(ToolArgumentModel):
    """Settle a returned outbound remittance with return FX and fee-refund handling."""

    remittance_id: str
    source_account_id: str
    return_reason: Literal["RECIPIENT_REJECTED_BY_CUSTOMER_INPUT"]
    bank_fault: bool
    expected_status: Literal["RETURNED_SETTLED"]
    original_exchange_rate_krw_per_unit: JsonNumber
    return_exchange_rate_krw_per_unit: JsonNumber
    original_principal_krw: JsonNumber
    returned_principal_krw: JsonNumber
    fx_loss_krw: JsonNumber
    send_fee_krw: JsonNumber
    send_fee_refunded: bool
    transaction_id: str | None = Field(
        description=(
            "New outbound return-settlement transaction ID. Set null to let the "
            "tool generate it. If supplied manually, for remittance_id "
            "remit_outbound_NNN use txn_outbound_return_settlement_NNN."
        )
    )

    @model_validator(mode="after")
    def validate_generated_ids(self):
        _validate_generated_id(
            source_id=self.remittance_id,
            field_name="transaction_id",
            actual=self.transaction_id,
            prefix="txn_outbound_return_settlement",
        )
        return self


RemittanceOptions = (
    DollarBoxGiftAutoCancelOptions
    | DollarBoxGiftReceiveOptions
    | InboundAutoReceiveDocumentRequestOptions
    | InboundDailyOver100kDocumentRequestOptions
    | InboundBulkDepositOptions
    | InboundReturnInfoMismatchOptions
    | InboundResidencyVerificationHoldOptions
    | InboundImmediateDepositOptions
    | OutboundBeneficiaryInfoAutoCancelOptions
    | OutboundBusinessPurposeRejectedOptions
    | OutboundNoDocumentOptions
    | OutboundNoDocumentOver100kSingleLimitRejectedOptions
    | OutboundReturnSettlementOptions
)


class ExecuteRemittanceCaseArguments(RemittanceCommonArguments):
    direction: Literal[
        "DOLLARBOX_GIFT_AUTO_CANCEL",
        "DOLLARBOX_GIFT_RECEIVE",
        "INBOUND_AUTO_RECEIVE_DOCUMENT_REQUEST",
        "INBOUND_DAILY_OVER_100K_DOCUMENT_REQUEST",
        "INBOUND_BULK_DEPOSIT",
        "INBOUND_RETURN_INFO_MISMATCH",
        "INBOUND_RESIDENCY_VERIFICATION_HOLD",
        "INBOUND_IMMEDIATE_DEPOSIT",
        "OUTBOUND_BENEFICIARY_INFO_AUTO_CANCEL",
        "OUTBOUND_BUSINESS_PURPOSE_REJECTED",
        "OUTBOUND_NO_DOCUMENT",
        "OUTBOUND_NO_DOCUMENT_OVER_100K_SINGLE_LIMIT_REJECTED",
        "OUTBOUND_RETURN_SETTLEMENT",
    ] = Field(
        description=(
            "Policy-grounded remittance mutation branch. Use "
            "INBOUND_AUTO_RECEIVE_DOCUMENT_REQUEST when an inbound remittance "
            "matches auto-receive but a personal-business account, purpose, or "
            "evidence rule requires review before deposit. Use "
            "INBOUND_IMMEDIATE_DEPOSIT only when policy allows immediate inbound "
            "deposit without receive application. Other values map to explicit "
            "DollarBox gift, inbound hold/return/bulk-deposit, outbound send, "
            "outbound reject, outbound auto-cancel, or outbound return-settlement "
            "flows."
        )
    )
    options: RemittanceOptions = Field(
        description=(
            "Direction-specific argument object. Its shape must match the chosen "
            "direction and contain the exact runtime IDs, amounts, fees, exchange "
            "rates, reasons, and status fields needed for the DB mutation."
        )
    )


class DollarBoxGiftAutoCancelArguments(RemittanceCommonArguments):
    direction: Literal["DOLLARBOX_GIFT_AUTO_CANCEL"]
    options: DollarBoxGiftAutoCancelOptions


class DollarBoxGiftReceiveArguments(RemittanceCommonArguments):
    direction: Literal["DOLLARBOX_GIFT_RECEIVE"]
    options: DollarBoxGiftReceiveOptions


class InboundAutoReceiveDocumentRequestArguments(RemittanceCommonArguments):
    direction: Literal["INBOUND_AUTO_RECEIVE_DOCUMENT_REQUEST"]
    options: InboundAutoReceiveDocumentRequestOptions


class InboundDailyOver100kDocumentRequestArguments(RemittanceCommonArguments):
    direction: Literal["INBOUND_DAILY_OVER_100K_DOCUMENT_REQUEST"]
    options: InboundDailyOver100kDocumentRequestOptions


class InboundBulkDepositArguments(RemittanceCommonArguments):
    direction: Literal["INBOUND_BULK_DEPOSIT"]
    options: InboundBulkDepositOptions


class InboundReturnInfoMismatchArguments(RemittanceCommonArguments):
    direction: Literal["INBOUND_RETURN_INFO_MISMATCH"]
    options: InboundReturnInfoMismatchOptions


class InboundResidencyVerificationHoldArguments(RemittanceCommonArguments):
    direction: Literal["INBOUND_RESIDENCY_VERIFICATION_HOLD"]
    options: InboundResidencyVerificationHoldOptions


class InboundImmediateDepositArguments(RemittanceCommonArguments):
    direction: Literal["INBOUND_IMMEDIATE_DEPOSIT"]
    options: InboundImmediateDepositOptions


class OutboundBeneficiaryInfoAutoCancelArguments(RemittanceCommonArguments):
    direction: Literal["OUTBOUND_BENEFICIARY_INFO_AUTO_CANCEL"]
    options: OutboundBeneficiaryInfoAutoCancelOptions


class OutboundBusinessPurposeRejectedArguments(RemittanceCommonArguments):
    direction: Literal["OUTBOUND_BUSINESS_PURPOSE_REJECTED"]
    options: OutboundBusinessPurposeRejectedOptions


class OutboundNoDocumentArguments(RemittanceCommonArguments):
    direction: Literal["OUTBOUND_NO_DOCUMENT"]
    options: OutboundNoDocumentOptions


class OutboundNoDocumentOver100kSingleLimitRejectedArguments(
    RemittanceCommonArguments
):
    direction: Literal["OUTBOUND_NO_DOCUMENT_OVER_100K_SINGLE_LIMIT_REJECTED"]
    options: OutboundNoDocumentOver100kSingleLimitRejectedOptions


class OutboundReturnSettlementArguments(RemittanceCommonArguments):
    direction: Literal["OUTBOUND_RETURN_SETTLEMENT"]
    options: OutboundReturnSettlementOptions


REMITTANCE_ARGUMENT_MODEL_BY_DIRECTION: dict[str, type[ToolArgumentModel]] = {
    "DOLLARBOX_GIFT_AUTO_CANCEL": DollarBoxGiftAutoCancelArguments,
    "DOLLARBOX_GIFT_RECEIVE": DollarBoxGiftReceiveArguments,
    "INBOUND_AUTO_RECEIVE_DOCUMENT_REQUEST": InboundAutoReceiveDocumentRequestArguments,
    "INBOUND_DAILY_OVER_100K_DOCUMENT_REQUEST": (
        InboundDailyOver100kDocumentRequestArguments
    ),
    "INBOUND_BULK_DEPOSIT": InboundBulkDepositArguments,
    "INBOUND_RETURN_INFO_MISMATCH": InboundReturnInfoMismatchArguments,
    "INBOUND_RESIDENCY_VERIFICATION_HOLD": InboundResidencyVerificationHoldArguments,
    "INBOUND_IMMEDIATE_DEPOSIT": InboundImmediateDepositArguments,
    "OUTBOUND_BENEFICIARY_INFO_AUTO_CANCEL": OutboundBeneficiaryInfoAutoCancelArguments,
    "OUTBOUND_BUSINESS_PURPOSE_REJECTED": OutboundBusinessPurposeRejectedArguments,
    "OUTBOUND_NO_DOCUMENT": OutboundNoDocumentArguments,
    "OUTBOUND_NO_DOCUMENT_OVER_100K_SINGLE_LIMIT_REJECTED": (
        OutboundNoDocumentOver100kSingleLimitRejectedArguments
    ),
    "OUTBOUND_RETURN_SETTLEMENT": OutboundReturnSettlementArguments,
}


PYDANTIC_TOOL_ARGUMENT_MODELS: dict[str, type[ToolArgumentModel]] = {
    "execute_remittance_case": ExecuteRemittanceCaseArguments,
}


def pydantic_tool_parameters(tool_name: str) -> dict[str, Any] | None:
    model = PYDANTIC_TOOL_ARGUMENT_MODELS.get(tool_name)
    if model is None:
        return None
    return _inline_local_refs(to_strict_json_schema(model))


def validate_pydantic_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> str | None:
    if tool_name != "execute_remittance_case":
        return None

    direction = arguments.get("direction")
    model = REMITTANCE_ARGUMENT_MODEL_BY_DIRECTION.get(str(direction))
    if model is None:
        return (
            "ToolArgumentValidationError: execute_remittance_case.direction must be "
            f"one of {', '.join(sorted(REMITTANCE_ARGUMENT_MODEL_BY_DIRECTION))}."
        )

    try:
        model.model_validate(arguments)
    except ValidationError as exc:
        return _format_validation_error(tool_name, exc)
    return None


def _format_validation_error(tool_name: str, exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:8]:
        loc = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {error.get('msg', 'invalid value')}")
    if len(exc.errors()) > 8:
        parts.append(f"... {len(exc.errors()) - 8} more")
    return f"ToolArgumentValidationError for {tool_name}: " + "; ".join(parts)


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                resolved = copy.deepcopy(defs[name])
                for key, value in node.items():
                    if key != "$ref":
                        resolved[key] = value
                return resolve(resolved)
            return {
                key: resolve(value)
                for key, value in node.items()
                if key != "$defs"
            }
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)
