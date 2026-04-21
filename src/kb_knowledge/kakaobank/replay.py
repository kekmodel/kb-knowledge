"""Minimal v0 DB replay helpers for KakaoBank candidate tasks.

This is intentionally not a full tau2 environment. It only covers the first
vertical slice: load an empty KakaoBank DB, apply per-task initialization data,
and replay gold actions far enough to verify read-only/refusal tasks.
Mutating tools stay explicit ``NotImplemented`` until their deterministic DB
postconditions are implemented.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb_knowledge.kakaobank.data_model import (
    ACTION_VERIFIER_SCHEMA_PATH,
    KakaoBankDB,
)

KAKAOBANK_KNOWLEDGE_DOMAIN_DIR = Path("data/kakaobank_knowledge/v0")
KAKAOBANK_KNOWLEDGE_DB_PATH = KAKAOBANK_KNOWLEDGE_DOMAIN_DIR / "tasks" / "db.json"
KAKAOBANK_KNOWLEDGE_TASKS_DIR = KAKAOBANK_KNOWLEDGE_DOMAIN_DIR / "tasks" / "cases"
CANONICAL_TOOL_TIMESTAMP = "TOOL_EXECUTION_TIME"
SERVER_TIMESTAMP_FIELDS = {
    "effective_at",
    "executed_at",
    "issued_at",
    "lost_reported_at",
    "processed_at",
    "receive_completed_at",
    "recipient_completed_at",
    "reconsented_at",
    "redeposited_at",
    "refunded_at",
    "reported_at",
    "requested_at",
    "restricted_at",
}


class ReplayError(RuntimeError):
    """Base error for v0 DB replay failures."""


class UnknownReplayActionError(ReplayError):
    """Raised when a task action is absent from the action schema."""


class ReplayRequestorMismatchError(ReplayError):
    """Raised when a task action requestor differs from the action schema."""


class UnsupportedMutatingActionError(ReplayError):
    """Raised for write tools before their deterministic replay exists."""


class ReplayTargetRecordNotFoundError(ReplayError):
    """Raised when a mutating replay action points at a missing DB record."""


@dataclass
class ReplayedAction:
    """One action considered during DB replay."""

    action_id: str
    name: str
    requestor: str
    mutates_state: bool
    status: str


@dataclass
class ReplayResult:
    """DB replay result for one task."""

    task_id: str
    initialized_hash: str
    final_hash: str
    actions: list[ReplayedAction]
    db: KakaoBankDB

    @property
    def unchanged_from_initialized(self) -> bool:
        """Return True when replay leaves the initialized DB unchanged."""

        return self.initialized_hash == self.final_hash


@dataclass
class DbEvaluationResult:
    """DB-only True/False result for candidate assistant actions."""

    task_id: str
    passed: bool
    initialized_hash: str
    expected_final_hash: str
    actual_final_hash: str | None
    error: str | None = None


def build_empty_domain_db() -> KakaoBankDB:
    """Build the empty runtime DB fixture with schema-derived table notes."""

    return KakaoBankDB.empty_with_schema_notes()


def write_empty_domain_db(path: Path = KAKAOBANK_KNOWLEDGE_DB_PATH) -> Path:
    """Write the empty runtime DB fixture and return its path."""

    db = build_empty_domain_db()
    db.dump(path)
    return path


def load_domain_db(path: Path = KAKAOBANK_KNOWLEDGE_DB_PATH) -> KakaoBankDB:
    """Load the KakaoBank runtime DB fixture."""

    return KakaoBankDB.load(path)


def load_exported_task(
    task_id: str,
    *,
    tasks_dir: Path = KAKAOBANK_KNOWLEDGE_TASKS_DIR,
) -> dict[str, Any]:
    """Load one exported v0 candidate task by ID."""

    path = tasks_dir / f"{task_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_initialization_data(
    db: KakaoBankDB,
    agent_data: dict[str, Any] | None,
) -> KakaoBankDB:
    """Return a DB with tau2-style ``agent_data`` overlaid onto it."""

    if not agent_data:
        return KakaoBankDB.model_validate(db.model_dump())

    merged = deepcopy(db.model_dump())
    _deep_merge(merged, agent_data)
    return KakaoBankDB.model_validate(merged)


def apply_task_initial_state(db: KakaoBankDB, task_data: dict[str, Any]) -> KakaoBankDB:
    """Apply one exported task's initial state to ``db``."""

    initial_state = task_data.get("initial_state")
    if not initial_state:
        return KakaoBankDB.model_validate(db.model_dump())

    initialization_data = initial_state.get("initialization_data") or {}
    agent_data = initialization_data.get("agent_data")
    return apply_initialization_data(db, agent_data)


def replay_expected_actions(
    task_data: dict[str, Any],
    *,
    db: KakaoBankDB | None = None,
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
) -> ReplayResult:
    """Replay a task's expected actions into a DB.

    Read-only actions are validated and recorded as skipped because they must
    not affect the DB hash. Supported mutating actions are applied
    deterministically; unsupported ones raise ``UnsupportedMutatingActionError``.
    """

    if db is None:
        db = build_empty_domain_db()
    initialized_db = apply_task_initial_state(db, task_data)
    initialized_hash = initialized_db.get_hash()

    schema_actions = _action_schema_by_name(schema_path)
    replayed: list[ReplayedAction] = []
    actions = task_data.get("evaluation_criteria", {}).get("actions") or []
    for index, action in enumerate(actions):
        replayed.append(
            replay_expected_action(
                initialized_db,
                action,
                schema_actions=schema_actions,
                task_id=str(task_data.get("id", "")),
                action_index=index,
            )
        )

    return ReplayResult(
        task_id=str(task_data["id"]),
        initialized_hash=initialized_hash,
        final_hash=initialized_db.get_hash(),
        actions=replayed,
        db=initialized_db,
    )


def replay_candidate_actions(
    task_data: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    db: KakaoBankDB | None = None,
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
) -> ReplayResult:
    """Replay candidate assistant actions into a task-initialized DB.

    Candidate replay mirrors live tool execution: a state-changing tool call
    that fails a runtime precondition returns an error and leaves DB state
    unchanged. Schema-level problems still raise because those calls could not
    be executed by the declared tool surface.
    """

    if db is None:
        db = build_empty_domain_db()
    initialized_db = apply_task_initial_state(db, task_data)
    initialized_hash = initialized_db.get_hash()

    schema_actions = _action_schema_by_name(schema_path)
    replayed: list[ReplayedAction] = []
    for index, action in enumerate(actions):
        try:
            replayed.append(
                replay_expected_action(
                    initialized_db,
                    action,
                    schema_actions=schema_actions,
                    task_id=str(task_data.get("id", "")),
                    action_index=index,
                )
            )
        except (
            UnknownReplayActionError,
            ReplayRequestorMismatchError,
            UnsupportedMutatingActionError,
        ):
            raise
        except ReplayError as exc:
            replayed.append(
                ReplayedAction(
                    action_id=str(
                        action.get("action_id", f"{task_data['id']}_{index:02d}")
                    ),
                    name=str(action.get("name", "")),
                    requestor=str(action.get("requestor", "assistant")),
                    mutates_state=True,
                    status=f"failed_no_state_change:{type(exc).__name__}: {exc}",
                )
            )

    return ReplayResult(
        task_id=str(task_data["id"]),
        initialized_hash=initialized_hash,
        final_hash=initialized_db.get_hash(),
        actions=replayed,
        db=initialized_db,
    )


def evaluate_candidate_actions(
    task_data: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    db: KakaoBankDB | None = None,
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
) -> DbEvaluationResult:
    """Return DB-only pass/fail by comparing candidate replay to gold replay."""

    if db is None:
        db = build_empty_domain_db()

    expected_db = KakaoBankDB.model_validate(db.model_dump())
    actual_db = KakaoBankDB.model_validate(db.model_dump())
    expected = replay_expected_actions(
        task_data,
        db=expected_db,
        schema_path=schema_path,
    )
    try:
        actual = replay_candidate_actions(
            task_data,
            actions,
            db=actual_db,
            schema_path=schema_path,
        )
    except ReplayError as exc:
        return DbEvaluationResult(
            task_id=str(task_data["id"]),
            passed=False,
            initialized_hash=expected.initialized_hash,
            expected_final_hash=expected.final_hash,
            actual_final_hash=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return DbEvaluationResult(
        task_id=str(task_data["id"]),
        passed=actual.final_hash == expected.final_hash,
        initialized_hash=expected.initialized_hash,
        expected_final_hash=expected.final_hash,
        actual_final_hash=actual.final_hash,
    )


def replay_expected_action(
    db: KakaoBankDB,
    action: dict[str, Any],
    *,
    schema_actions: dict[str, dict[str, Any]] | None = None,
    task_id: str = "",
    action_index: int = 0,
) -> ReplayedAction:
    """Replay one expected action against ``db``."""

    if schema_actions is None:
        schema_actions = _action_schema_by_name()

    action_name = str(action.get("name", ""))
    schema_action = schema_actions.get(action_name)
    if schema_action is None:
        raise UnknownReplayActionError(
            f"{task_id}: unknown action at index {action_index}: {action_name!r}"
        )

    requestor = str(action.get("requestor", "assistant"))
    expected_requestor = schema_action["requestor"]
    if requestor != expected_requestor:
        raise ReplayRequestorMismatchError(
            f"{task_id}: action {action_name} requestor {requestor!r} "
            f"does not match schema {expected_requestor!r}"
        )

    mutates_state = bool(schema_action["mutates_state"])
    if mutates_state:
        status = replay_mutating_expected_action(
            db,
            action,
            task_id=task_id,
            action_index=action_index,
        )
        return ReplayedAction(
            action_id=str(action.get("action_id", f"{task_id}_{action_index:02d}")),
            name=action_name,
            requestor=requestor,
            mutates_state=True,
            status=status,
        )

    # Read-only tools are intentionally not executed during DB replay.
    return ReplayedAction(
        action_id=str(action.get("action_id", f"{task_id}_{action_index:02d}")),
        name=action_name,
        requestor=requestor,
        mutates_state=False,
        status="skipped_read_only",
    )


def replay_mutating_expected_action(
    db: KakaoBankDB,
    action: dict[str, Any],
    *,
    task_id: str = "",
    action_index: int = 0,
) -> str:
    """Apply one supported mutating expected action and return replay status."""

    action_name = str(action.get("name", ""))
    if action_name == "close_account_or_service":
        return _replay_close_account_or_service(db, action)
    if action_name == "execute_deposit_or_box_transfer":
        return _replay_execute_deposit_or_box_transfer(db, action)
    if action_name == "open_or_enroll_product":
        return _replay_open_or_enroll_product(db, action)
    if action_name == "update_loan_contract_state":
        return _replay_update_loan_contract_state(db, action)
    if action_name == "request_maturity_or_extension":
        return _replay_request_maturity_or_extension(db, action)
    if action_name == "execute_remittance_case":
        return _replay_execute_remittance_case(db, action)
    if action_name == "update_card_state":
        return _replay_update_card_state(db, action)
    if action_name == "file_dispute_or_objection":
        return _replay_file_dispute_or_objection(db, action)
    if action_name == "process_refinance_request":
        return _replay_process_refinance_request(db, action)
    if action_name == "create_loan_application":
        return _replay_create_loan_application(db, action)
    if action_name == "configure_auto_transfer":
        return _replay_configure_auto_transfer(db, action)
    if action_name == "request_interest_payment":
        return _replay_request_interest_payment(db, action)

    raise UnsupportedMutatingActionError(
        f"{task_id}: mutating action {action_name!r} at index {action_index} "
        "does not have deterministic v0 replay implemented yet"
    )


def _action_schema_by_name(
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
) -> dict[str, dict[str, Any]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in schema["action_families"]}


def _replay_close_account_or_service(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    target_id = str(arguments["target_id"])
    close_type = str(arguments["close_type"])
    options = arguments.get("options") or {}

    if close_type.startswith("REJECT"):
        return "rejected_noop"

    _, target_record = _find_record_by_id(db, target_id)
    if close_type == "REMOVE_GROUP_MEMBER":
        target_record["status"] = str(options.get("new_membership_status", "REMOVED"))
        return "applied_close_account_or_service"

    target_record["status"] = "CLOSED"

    if close_type == "CHILD_ACCOUNT_SELF_CLOSE_AT_AGE_19":
        _transfer_krw_balance(
            db,
            source_record=target_record,
            target_id=str(arguments["destination_account_id"]),
            amount=_numeric_value(target_record.get("balance_krw")),
            source_balance_field="balance_krw",
        )
        also_close_service_id = arguments.get("also_close_service_id")
        if also_close_service_id:
            _set_record_status(db, str(also_close_service_id), "CLOSED")

    if close_type in {"CLOSE_SAFEBOX", "CLOSE_PIGGY_BANK", "CLOSE_VAT_BOX"}:
        _transfer_box_balance_to_base_account(db, target_record, options)

    if close_type == "CLOSE_LAST_NON_INTEREST_SECTION_AND_RECORD_ACCOUNT":
        _close_record_book_section(db, target_record)

    if close_type == "CLOSE_GROUP_ACCOUNT_SERVICE":
        _convert_group_account_service(db, target_record, options)

    return "applied_close_account_or_service"


def _replay_execute_deposit_or_box_transfer(
    db: KakaoBankDB, action: dict[str, Any]
) -> str:
    arguments = action.get("arguments") or {}
    transfer_type = str(
        arguments.get("transaction_type") or arguments.get("transfer_type") or ""
    )

    if _is_rejected_transfer(arguments, transfer_type):
        return "rejected_transfer_noop"

    if arguments.get("credited_to_wallet") is False:
        _apply_transaction_record_updates(db, arguments)
        return "pending_transfer_noop"

    source_id = arguments.get("source_id") or arguments.get("source_account_id")
    target_id = arguments.get("target_id")
    amount = _numeric_value(arguments.get("amount"))

    source = _find_record_by_id_optional(db, str(source_id)) if source_id else None
    target = _find_record_by_id_optional(db, str(target_id)) if target_id else None
    if source is None and target is None:
        raise ReplayTargetRecordNotFoundError(
            f"transfer has no DB source or target record: {source_id!r} -> {target_id!r}"
        )

    if source is not None:
        _debit_record_if_balance_backed(source[1], amount)
    if target is not None:
        _credit_record_if_balance_backed(target[1], amount)

    _apply_transfer_record_updates(db, arguments, source=source, target=target)
    return "applied_deposit_or_box_transfer"


def _replay_open_or_enroll_product(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    operation = str(options.get("operation") or options.get("opening_mode") or "")

    if _is_rejected_open_operation(options, operation):
        return "rejected_open_noop"

    if operation in {"NEW_LIMIT_ACCOUNT", "OPEN_CHILD_ACCOUNT"}:
        _open_account_record(db, arguments, options)
        if operation == "OPEN_CHILD_ACCOUNT":
            _open_service_record(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation == "CONVERT_LIMIT_TO_NORMAL_ACCOUNT":
        _convert_limit_account(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation in {"ENROLL_SERVICE", "REENROLL_SERVICE"}:
        _open_service_record(db, arguments, options)
        _open_consent_records(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation == "NEW_PREPAID_WALLET":
        _open_prepaid_wallet_record(db, arguments, options)
        _open_consent_records(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation in {
        "OPEN_SAFEBOX",
        "OPEN_PIGGY_BANK",
        "OPEN_VAT_BOX",
        "OPEN_DOLLARBOX",
    }:
        _open_savings_box_record(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation == "NEW_FIXED_DEPOSIT":
        _open_deposit_contract_record(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation == "OPEN_GROUP_ACCOUNT_SERVICE":
        _open_service_record(db, arguments, options)
        _open_group_owner_membership(db, arguments, options)
        _link_service_to_account(db, arguments, options)
        return "applied_open_or_enroll_product"

    if operation == "RESTRICT_GROUP_ACCOUNT_SERVICE":
        _restrict_group_account_service(db, options)
        return "applied_open_or_enroll_product"

    if operation == "REACTIVATE_GROUP_ACCOUNT_SERVICE_AFTER_RECONSENT":
        _reactivate_group_account_service(db, options)
        return "applied_open_or_enroll_product"

    if operation == "CONNECTED_NEW":
        _open_record_book_account(db, arguments, options)
        return "applied_open_or_enroll_product"

    raise UnsupportedMutatingActionError(
        f"open_or_enroll_product operation {operation!r} is not implemented"
    )


def _is_rejected_open_operation(options: dict[str, Any], operation: str) -> bool:
    if operation.startswith("REJECT"):
        return True
    if options.get("expected_status") == "REJECTED":
        return True
    rejected_creation_flags = (
        "new_service_created",
        "new_box_created",
        "new_wallet_created",
        "feature_access_granted",
    )
    return any(options.get(flag) is False for flag in rejected_creation_flags)


def _open_account_record(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    account_id = str(options.get("account_id") or options["new_account_id"])
    is_limit_account = bool(options.get("is_limit_account", False))
    status = str(options.get("expected_status") or "ACTIVE")
    if is_limit_account and status == "ACTIVE":
        status = "ACTIVE_LIMIT_ACCOUNT"

    db.accounts.data[account_id] = {
        "account_id": account_id,
        "customer_id": arguments["customer_id"],
        "product_name": arguments["product_name"],
        "currency": options.get("currency", "KRW"),
        "balance_krw": options.get("opening_amount_krw", 0),
        "status": status,
        "is_limit_account": is_limit_account,
        "linked_service_ids": [],
        "restriction_flags": [],
    }
    _copy_option_fields(
        db.accounts.data[account_id],
        options,
        (
            "business_id",
            "business_name_display",
            "daily_app_transfer_limit_krw",
            "monthly_app_transfer_limit_krw",
            "financial_purpose_verified",
            "legal_guardian_consent_required_for_limit_release",
            "minor_account_holder",
            "requested_immediate_transfer_status",
        ),
    )


def _convert_limit_account(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    _, account = _find_record_by_id(db, str(arguments["source_account_id"]))
    account["status"] = str(options.get("expected_status", "ACTIVE"))
    account["is_limit_account"] = bool(options.get("is_limit_account", False))
    _copy_option_fields(
        account,
        options,
        (
            "daily_app_transfer_limit_krw",
            "financial_purpose_verified",
            "verification_method",
        ),
    )


def _open_service_record(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    service_id = options.get("service_id")
    if not service_id:
        return

    record = {
        "service_id": service_id,
        "customer_id": arguments["customer_id"],
        "service_name": arguments["product_name"],
        "status": str(options.get("expected_status", "ACTIVE")),
        "linked_account_id": arguments.get("source_account_id"),
        "restriction_flags": [],
    }
    _copy_option_fields(
        record,
        options,
        (
            "business_id",
            "provided_services",
            "previous_service_id",
            "financial_info_consent_status",
            "member_count",
            "terms_consent_status",
            "requested_by_customer_id",
        ),
    )
    db.service_enrollments.data[str(service_id)] = record


def _open_consent_records(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    for key, value in options.items():
        if not key.endswith("_consent_id") or not value:
            continue
        consent_id = str(value)
        db.consents.data[consent_id] = {
            "consent_id": consent_id,
            "customer_id": arguments["customer_id"],
            "consent_type": key.removesuffix("_id").upper(),
            "status": "ACTIVE",
            "valid": True,
        }


def _open_prepaid_wallet_record(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    wallet_id = str(options["wallet_id"])
    db.prepaid_wallets.data[wallet_id] = {
        "wallet_id": wallet_id,
        "customer_id": arguments["customer_id"],
        "product_name": arguments["product_name"],
        "status": "ACTIVE",
        "balance_krw": 0,
        "age_band": options.get("age_band"),
        "holding_limit_krw": options.get("holding_limit_krw"),
        "daily_limit_krw": options.get("daily_spend_limit_krw"),
        "monthly_limit_krw": options.get("monthly_spend_limit_krw"),
        "legal_form": options.get("legal_form"),
    }


def _open_savings_box_record(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    box_id = str(options["box_id"])
    balance = options.get("initial_balance", options.get("opening_amount_krw", 0))
    record = {
        "box_id": box_id,
        "customer_id": arguments["customer_id"],
        "product_name": arguments["product_name"],
        "status": "ACTIVE",
        "currency": options.get("currency", "KRW"),
        "balance": balance,
        "base_account_id": options.get("base_account_id")
        or arguments.get("source_account_id"),
        "limit_amount": options.get("limit_amount"),
        "restriction_flags": [],
    }
    _copy_option_fields(
        record,
        options,
        (
            "business_id",
            "business_number",
            "base_account_changeable",
            "limit_increased",
            "terms_consent_status",
        ),
    )
    db.savings_boxes.data[box_id] = record


def _open_deposit_contract_record(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    deposit_id = str(options["deposit_id"])
    principal = _numeric_value(options.get("principal_krw"))
    db.deposit_contracts.data[deposit_id] = {
        "deposit_id": deposit_id,
        "customer_id": arguments["customer_id"],
        "product_name": arguments["product_name"],
        "status": "ACTIVE",
        "currency": options.get("currency", "KRW"),
        "base_account_id": options.get("base_account_id")
        or arguments.get("source_account_id"),
        "principal_krw": principal,
        "term_months": options.get("term_months"),
        "tax_free_savings": options.get("tax_free_savings"),
        "maturity_management": options.get("maturity_management"),
        "coupon_id": options.get("coupon_id"),
    }
    source_account_id = arguments.get("source_account_id")
    if source_account_id:
        source = _find_record_by_id_optional(db, str(source_account_id))
        if source is not None:
            _debit_record_if_balance_backed(source[1], principal)


def _open_group_owner_membership(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    membership_id = options.get("owner_membership_id")
    if not membership_id:
        return

    db.group_memberships.data[str(membership_id)] = {
        "membership_id": membership_id,
        "service_id": options.get("service_id"),
        "account_id": arguments.get("source_account_id"),
        "customer_id": arguments["customer_id"],
        "role": "OWNER",
        "status": "ACTIVE",
        "financial_info_consent_status": options.get("financial_info_consent_status"),
    }


def _link_service_to_account(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    source_account_id = arguments.get("source_account_id")
    service_id = options.get("service_id")
    if not source_account_id or not service_id:
        return

    account = _find_record_by_id_optional(db, str(source_account_id))
    if account is None:
        return

    record = account[1]
    linked_service_ids = record.setdefault("linked_service_ids", [])
    if service_id not in linked_service_ids:
        linked_service_ids.append(service_id)
    if options.get("apply_limit_loan_restriction"):
        restriction_flags = record.setdefault("restriction_flags", [])
        if "LIMIT_LOAN_BLOCKED_BY_GROUP_ACCOUNT" not in restriction_flags:
            restriction_flags.append("LIMIT_LOAN_BLOCKED_BY_GROUP_ACCOUNT")


def _restrict_group_account_service(db: KakaoBankDB, options: dict[str, Any]) -> None:
    _, service = _find_record_by_id(db, str(options["service_id"]))
    service["status"] = "RESTRICTED"
    restriction_flags = service.setdefault("restriction_flags", [])
    if "FINANCIAL_INFO_RECONSENT_OVERDUE" not in restriction_flags:
        restriction_flags.append("FINANCIAL_INFO_RECONSENT_OVERDUE")


def _reactivate_group_account_service(db: KakaoBankDB, options: dict[str, Any]) -> None:
    _, service = _find_record_by_id(db, str(options["service_id"]))
    service["status"] = "ACTIVE"
    service["financial_info_consent_status"] = options.get(
        "financial_info_consent_status", "ACTIVE"
    )
    if "reconsented_at" in options:
        service["reconsented_at"] = _tool_timestamp()

    cleared = set(options.get("restriction_flags_cleared") or [])
    if isinstance(service.get("restriction_flags"), list):
        service["restriction_flags"] = [
            flag for flag in service["restriction_flags"] if flag not in cleared
        ]

    owner_membership_id = options.get("owner_membership_id")
    if owner_membership_id:
        _set_record_status(db, str(owner_membership_id), "ACTIVE")


def _open_record_book_account(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    account_id = str(options["account_id"])
    base_account_id = str(options["base_account_id"])
    first_section = options["first_section"]
    pocket_id = str(first_section["pocket_id"])

    db.accounts.data[account_id] = {
        "account_id": account_id,
        "customer_id": arguments["customer_id"],
        "product_name": arguments["product_name"],
        "status": "ACTIVE",
        "currency": "KRW",
        "balance_krw": options.get("initial_amount_krw", 0),
        "linked_account_id": base_account_id,
        "non_interest_section_count": 1,
        "restriction_flags": [],
    }
    db.pockets.data[pocket_id] = {
        "pocket_id": pocket_id,
        "parent_id": account_id,
        "pocket_type": "RECORD_BOOK_SECTION",
        "section_name": first_section["section_name"],
        "status": "ACTIVE",
        "balance_krw": options.get("initial_amount_krw", 0),
        "is_interest_section": False,
        "collection_rules": first_section.get("collection_rules", []),
    }


def _copy_option_fields(
    target: dict[str, Any],
    options: dict[str, Any],
    field_names: tuple[str, ...],
) -> None:
    for field_name in field_names:
        if field_name in options:
            target[field_name] = _canonicalize_replay_value(
                field_name,
                options[field_name],
            )


def _canonicalize_replay_value(field_name: str, value: Any) -> Any:
    if field_name in SERVER_TIMESTAMP_FIELDS and value is not None:
        return CANONICAL_TOOL_TIMESTAMP
    return value


def _tool_timestamp() -> str:
    return CANONICAL_TOOL_TIMESTAMP


def _replay_request_maturity_or_extension(
    db: KakaoBankDB, action: dict[str, Any]
) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    operation = str(arguments["operation"])
    target_id = str(arguments["target_id"])

    if operation.startswith("REJECT"):
        return "rejected_maturity_noop"

    _, deposit = _find_record_by_id(db, target_id)
    if operation in {
        "MATURE_DIRECT_CLOSE",
        "MATURE_HOLIDAY_PREVIOUS_BUSINESS_DAY_CLOSE",
        "MATURE_AUTO_CLOSE",
        "EARLY_CLOSE",
    }:
        _close_deposit_contract_for_maturity(db, deposit, arguments, options)
        return "applied_maturity_or_extension"

    if operation == "AUTO_EXTEND":
        _auto_extend_deposit_contract(deposit, arguments, options)
        return "applied_maturity_or_extension"

    if operation == "AUTO_REDEPOSIT":
        _auto_redeposit_contract(deposit, arguments, options)
        return "applied_maturity_or_extension"

    raise UnsupportedMutatingActionError(
        f"request_maturity_or_extension operation {operation!r} is not implemented"
    )


def _close_deposit_contract_for_maturity(
    db: KakaoBankDB,
    deposit: dict[str, Any],
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    deposit["status"] = "CLOSED"
    deposit["close_type"] = str(options["close_type"])
    deposit["closed_at"] = _tool_timestamp()
    _copy_option_fields(
        deposit,
        options,
        (
            "maturity_payout_krw",
            "early_close_payout_krw",
            "preferential_rate_points",
            "preferential_rate_points_applied",
            "coupon_rate_applied",
            "auto_extended_principal_preferential_rate_points",
            "rejected_reason",
            "interest_days_policy",
            "close_date",
            "contractual_maturity_date",
        ),
    )
    if "preferential_rate_points" in options:
        deposit["preferential_rate_points_applied"] = options[
            "preferential_rate_points"
        ]

    destination_account_id = str(options["destination_account_id"])
    deposit["destination_account_id"] = destination_account_id
    payout = _first_numeric_option(
        options,
        ("maturity_payout_krw", "early_close_payout_krw"),
    )
    balance_field = _balance_field(deposit)
    if balance_field is None:
        raise ReplayTargetRecordNotFoundError(
            f"deposit contract has no balance-backed field: {deposit!r}"
        )
    _transfer_krw_balance(
        db,
        source_record=deposit,
        target_id=destination_account_id,
        amount=payout,
        source_balance_field=balance_field,
    )


def _auto_extend_deposit_contract(
    deposit: dict[str, Any],
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    deposit["status"] = "ACTIVE_EXTENDED"
    deposit["close_type"] = "AUTO_EXTENSION"
    deposit["extension_reason"] = options.get("reason")
    deposit["maturity_reason"] = options.get("reason")
    deposit["extended_at"] = _tool_timestamp()
    if "new_auto_extension_count" in options:
        deposit["auto_extension_count"] = options["new_auto_extension_count"]
    if "same_contract_months" in options:
        deposit["contract_months"] = options["same_contract_months"]


def _auto_redeposit_contract(
    deposit: dict[str, Any],
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    deposit["status"] = "ACTIVE_REDEPOSITED"
    deposit["close_type"] = "AUTO_REDEPOSIT"
    deposit["redeposit_reason"] = options.get("reason")
    deposit["maturity_reason"] = options.get("reason")
    deposit["redeposited_at"] = _tool_timestamp()
    if "principal_plus_interest_krw" in options:
        deposit["principal_krw"] = options["principal_plus_interest_krw"]
    if "accrued_interest_krw" in deposit:
        deposit["accrued_interest_krw"] = 0
    if "new_contract_years" in options:
        deposit["contract_years"] = options["new_contract_years"]


def _replay_execute_remittance_case(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    direction = str(arguments["direction"])

    if (
        direction == "DOLLARBOX_PENDING_RETURN_MANUAL_PROCESS"
        and options.get("manual_process_approved") is False
    ):
        return "rejected_remittance_noop"

    if direction.startswith("DOLLARBOX_GIFT_"):
        _replay_dollarbox_gift_remittance(db, arguments, options, direction)
        return "applied_remittance_case"

    if direction.startswith("INBOUND_"):
        _replay_inbound_remittance(db, arguments, options, direction)
        return "applied_remittance_case"

    if direction.startswith("OUTBOUND_"):
        _replay_outbound_remittance(db, arguments, options, direction)
        return "applied_remittance_case"

    raise UnsupportedMutatingActionError(
        f"execute_remittance_case direction {direction!r} is not implemented"
    )


def _replay_dollarbox_gift_remittance(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
    direction: str,
) -> None:
    remittance = _get_or_create_remittance_case(db, arguments, options)
    amount = _numeric_value(arguments["amount"])

    if direction == "DOLLARBOX_GIFT_RECEIVE":
        transaction_id = _generated_remittance_transaction_id(
            direction, options, "receive_transaction_id"
        )
        remittance["status"] = "RECEIVED"
        remittance["recipient_completed_at"] = _tool_timestamp()
        remittance["recipient_real_name_confirmed"] = options.get(
            "recipient_real_name_confirmed", True
        )
        remittance["receive_transaction_id"] = transaction_id
        recipient_box_id = str(options["recipient_box_id"])
        _, recipient_box = _find_record_by_id(db, recipient_box_id)
        _credit_record_if_balance_backed(recipient_box, amount)
        _remove_pending_id(
            recipient_box, "pending_inbound_gift_ids", remittance["remittance_id"]
        )
        _upsert_transaction(
            db,
            transaction_id=transaction_id,
            record_id=recipient_box_id,
            amount=amount,
            currency=str(arguments["currency"]),
            transaction_type="DOLLARBOX_GIFT_RECEIVE",
        )
        return

    if direction == "DOLLARBOX_GIFT_AUTO_CANCEL":
        transaction_id = _generated_remittance_transaction_id(
            direction, options, "refund_transaction_id"
        )
        remittance["status"] = "CANCELLED_REFUNDED"
        remittance["cancel_reason"] = options.get("cancel_reason")
        remittance["refund_transaction_id"] = transaction_id
        sender_box_id = str(options["sender_box_id"])
        _, sender_box = _find_record_by_id(db, sender_box_id)
        _credit_record_if_balance_backed(sender_box, amount)
        _remove_pending_id(
            sender_box, "pending_outbound_gift_ids", remittance["remittance_id"]
        )

        recipient_box = _find_record_by_id_optional(
            db, str(options["recipient_box_id"])
        )
        if recipient_box is not None:
            _remove_pending_id(
                recipient_box[1],
                "pending_inbound_gift_ids",
                remittance["remittance_id"],
            )
        _upsert_transaction(
            db,
            transaction_id=transaction_id,
            record_id=sender_box_id,
            amount=amount,
            currency=str(arguments["currency"]),
            transaction_type="DOLLARBOX_GIFT_REFUND",
        )
        return

    raise UnsupportedMutatingActionError(
        f"dollarbox remittance direction {direction!r} is not implemented"
    )


def _replay_inbound_remittance(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
    direction: str,
) -> None:
    remittance = _get_or_create_remittance_case(db, arguments, options)
    expected_status = str(options.get("expected_status", "DEPOSITED"))

    if direction in {"INBOUND_IMMEDIATE_DEPOSIT", "INBOUND_BULK_DEPOSIT"}:
        transaction_id = _generated_remittance_transaction_id(
            direction, options, "transaction_id"
        )
        remittance["status"] = "DEPOSITED"
        remittance["deposit_date"] = options.get("deposit_date")
        remittance["deposited_transaction_id"] = transaction_id
        remittance["receive_fee_krw"] = options.get("receive_fee_krw", 0)
        remittance["fee_waiver_reason"] = options.get("fee_waiver_reason")
        remittance["applied_exchange_rate_krw_per_unit"] = options.get(
            "exchange_rate_krw_per_unit"
        )
        remittance["expected_credit_amount_krw"] = options.get("credit_amount_krw")
        target_account_id = str(options["target_account_id"])
        _, target_account = _find_record_by_id(db, target_account_id)
        credit_amount = _numeric_value(options["credit_amount_krw"])
        _credit_record_if_balance_backed(target_account, credit_amount)
        _upsert_transaction(
            db,
            transaction_id=transaction_id,
            record_id=target_account_id,
            amount=credit_amount,
            currency="KRW",
            transaction_type="INBOUND_REMITTANCE_CREDIT",
        )
        _increment_remittance_profile_amount(
            db,
            customer_id=str(arguments["customer_id"]),
            field_name="annual_usd_received",
            amount=_numeric_value(arguments["amount"]),
        )
        return

    if direction == "INBOUND_RETURN_INFO_MISMATCH":
        options = dict(options)
        options["return_transaction_id"] = _generated_remittance_transaction_id(
            direction, options, "return_transaction_id"
        )
    remittance["status"] = expected_status
    _copy_remittance_options(
        remittance,
        options,
        (
            "document_request_reason",
            "daily_aggregate_usd_after_case",
            "mismatch_review_result",
            "return_transaction_id",
            "hold_reason",
            "resident_verified",
            "deposit_first_refused",
        ),
    )


def _replay_outbound_remittance(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
    direction: str,
) -> None:
    remittance = _get_or_create_remittance_case(db, arguments, options)

    if direction == "OUTBOUND_NO_DOCUMENT":
        transaction_id = _generated_remittance_transaction_id(
            direction, options, "transaction_id"
        )
        remittance["status"] = "SENT"
        remittance["debit_transaction_id"] = transaction_id
        _copy_remittance_options(
            remittance,
            options,
            (
                "recipient_name",
                "recipient_country",
                "recipient_relationship",
                "applied_exchange_rate_krw_per_unit",
                "fx_preference_rate_percent",
                "remittance_amount_krw",
                "send_fee_krw",
                "total_debit_krw",
                "wire_fee_waived",
                "intermediary_and_recipient_fee_borne_by",
                "annual_usd_equivalent",
            ),
        )
        source_account_id = str(options["source_account_id"])
        _, source_account = _find_record_by_id(db, source_account_id)
        total_debit = _numeric_value(options["total_debit_krw"])
        _debit_record_if_balance_backed(source_account, total_debit)
        _upsert_transaction(
            db,
            transaction_id=transaction_id,
            record_id=source_account_id,
            amount=-total_debit,
            currency="KRW",
            transaction_type="OUTBOUND_REMITTANCE_DEBIT",
        )
        _set_remittance_profile_amount(
            db,
            customer_id=str(arguments["customer_id"]),
            field_name="annual_usd_sent",
            amount=options.get("new_annual_usd_sent"),
        )
        return

    if options.get("expected_status") == "REJECTED":
        remittance["status"] = "REJECTED"
        remittance["rejection_reason"] = options.get("rejection_reason")
        _copy_remittance_options(
            remittance,
            options,
            (
                "source_account_id",
                "requested_sender_name",
                "business_account_as_source",
                "annual_usd_sent_before_case",
                "requested_no_document_amount_usd",
                "allowed_single_case_limit_usd_after_100k",
            ),
        )
        return

    if direction in {
        "OUTBOUND_BENEFICIARY_INFO_AUTO_CANCEL",
        "OUTBOUND_RETURN_SETTLEMENT",
    }:
        transaction_id = _generated_remittance_transaction_id(
            direction, options, "transaction_id"
        )
        remittance["status"] = str(options["expected_status"])
        remittance["returned_transaction_id"] = transaction_id
        remittance["send_fee_refunded"] = options.get("send_fee_refunded", False)
        _copy_remittance_options(
            remittance,
            options,
            (
                "processed_at",
                "return_reason",
                "bank_fault",
                "return_exchange_rate_krw_per_unit",
                "returned_principal_krw",
                "original_principal_krw",
                "fx_loss_krw",
                "send_fee_krw",
            ),
        )
        source_account_id = str(options["source_account_id"])
        _, source_account = _find_record_by_id(db, source_account_id)
        returned_principal = _numeric_value(options["returned_principal_krw"])
        _credit_record_if_balance_backed(source_account, returned_principal)
        _upsert_transaction(
            db,
            transaction_id=transaction_id,
            record_id=source_account_id,
            amount=returned_principal,
            currency="KRW",
            transaction_type="OUTBOUND_REMITTANCE_RETURN",
        )
        return

    raise UnsupportedMutatingActionError(
        f"outbound remittance direction {direction!r} is not implemented"
    )


def _replay_update_card_state(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    operation = str(arguments["operation"])

    if operation == "REJECT_PAYMENT":
        return "rejected_card_payment_noop"

    if operation == "REJECT_NEW_ISSUE":
        _upsert_card_order(db, arguments, status="REJECTED")
        return "rejected_card_order"

    if operation == "ISSUE_NEW_CARD":
        _issue_new_card(db, arguments)
        _upsert_card_order(db, arguments, status="APPROVED")
        return "applied_card_state"

    if operation == "REISSUE_CARD":
        _reissue_card(db, arguments)
        _upsert_card_order(db, arguments, status="APPROVED")
        return "applied_card_state"

    if operation == "REISSUE_CARD_WITHOUT_TMONEY_TRANSFER":
        _reissue_card_without_tmoney_transfer(db, arguments)
        _upsert_card_order(db, arguments, status="APPROVED")
        return "applied_card_state"

    if operation == "REPORT_LOST_CARD":
        _, card = _find_record_by_id(db, str(arguments["card_id"]))
        card["status"] = str(arguments.get("new_status", "LOST_REPORTED"))
        card["lost_reported_at"] = _tool_timestamp()
        card["lost_report_reason"] = arguments.get("reason")
        return "applied_card_state"

    if operation == "RESTRICT_CARD_AND_REJECT_TRANSACTION":
        _, card = _find_record_by_id(db, str(arguments["card_id"]))
        card["status"] = str(arguments.get("new_card_status", "RESTRICTED"))
        card["restricted_at"] = _tool_timestamp()
        card["restriction_reason"] = arguments.get("reason")
        return "applied_card_state"

    raise UnsupportedMutatingActionError(
        f"update_card_state operation {operation!r} is not implemented"
    )


def _replay_file_dispute_or_objection(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    target_id = str(arguments["target_id"])
    reason = str(arguments["reason"])
    dispute_id = _dispute_id(arguments)

    if reason == "LOST_CARD_COMPENSATION_ELIGIBLE_WITHIN_60_DAYS":
        card_id = options.get("card_id")
        if not card_id:
            raise ReplayError("lost-card compensation dispute requires card_id")
        _, card = _find_record_by_id(db, str(card_id))
        if card.get("status") != "LOST_REPORTED" or not card.get("lost_reported_at"):
            raise ReplayError(
                "lost-card compensation dispute requires a recorded lost-card report first"
            )

    dispute = {
        "dispute_id": dispute_id,
        "customer_id": arguments["customer_id"],
        "target_type": arguments["target_type"],
        "target_id": target_id,
        "reason": reason,
        "status": _dispute_status(options),
    }
    _copy_option_fields(
        dispute,
        options,
        (
            "investigation_status",
            "member_fault_flags",
            "compensation_approved",
            "liability_scope",
            "reported_at",
            "transaction_occurred_at",
            "within_60_days",
            "affiliate_responsibility",
            "contested_feature",
            "service_access_allowed",
            "reactivation_approved",
            "customer_fault_flags",
            "bank_or_nice_fault_found",
        ),
    )
    db.disputes.data[dispute_id] = dispute

    target = _find_record_by_id_optional(db, target_id)
    if target is not None and target[0] == "transactions":
        target[1]["disputed"] = True
        target[1]["dispute_id"] = dispute_id

    return "applied_dispute_or_objection"


def _replay_process_refinance_request(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    operation = str(arguments["operation"])
    refinance_id = str(arguments["refinance_id"])
    _, refinance = _find_record_by_id(db, refinance_id)

    refinance["old_loan_repayment_status"] = arguments.get("old_loan_repayment_status")
    refinance["processed_at"] = _tool_timestamp()

    if operation == "REJECT_UNREPAYABLE_OLD_LOAN":
        refinance["status"] = "REJECTED"
        refinance["rejection_reason"] = options.get("reason")
        refinance["cancellation_reason"] = options.get("reason")
        _copy_option_fields(refinance, options, ("home_type",))
        return "rejected_refinance_request"

    if operation == "CANCEL_NEW_LOAN_AFTER_OLD_REPAYMENT_FAILURE":
        refinance["status"] = "CANCELLED"
        refinance["cancellation_reason"] = "OLD_LOAN_REPAYMENT_FAILED"
        _set_refinance_new_loan_status(db, refinance, "CANCELLED")
        return "applied_refinance_request"

    if operation in {
        "COMPLETE_OLD_LOAN_REPAYMENT_AND_ACTIVATE_NEW_LOAN",
        "COMPLETE_OLD_LOAN_REPAYMENT_AND_REQUEST_LIEN_RELEASE",
    }:
        refinance["status"] = "COMPLETED"
        _close_refinanced_old_loan(db, refinance)
        _set_refinance_new_loan_status(
            db, refinance, str(options.get("new_loan_status_after", "ACTIVE"))
        )
        if operation == "COMPLETE_OLD_LOAN_REPAYMENT_AND_REQUEST_LIEN_RELEASE":
            refinance["lien_release_status"] = options.get(
                "lien_release_status", "REQUESTED"
            )
            _request_lien_release(db, refinance)
        return "applied_refinance_request"

    raise UnsupportedMutatingActionError(
        f"process_refinance_request operation {operation!r} is not implemented"
    )


def _replay_create_loan_application(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    application_id = str(arguments["application_id"])
    db.loan_applications.data[application_id] = {
        "application_id": application_id,
        "customer_id": arguments["customer_id"],
        "product_name": arguments["product_name"],
        "requested_amount_krw": arguments["requested_amount_krw"],
        "purpose": arguments["purpose"],
        "partner_id": arguments.get("partner_id"),
        "comparison_id": arguments.get("comparison_id"),
        "status": str(arguments["expected_status"]),
        "decision_reason": arguments.get("expected_reason")
        or _default_application_decision_reason(arguments),
    }
    return "applied_loan_application"


def _default_application_decision_reason(arguments: dict[str, Any]) -> str | None:
    if arguments.get("comparison_id"):
        return "QUOTE_NON_FINAL_PARTNER_CONTRACT_REQUIRED"
    return None


def _replay_configure_auto_transfer(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    operation = str(arguments["operation"])

    if operation.startswith("REJECT"):
        return "rejected_auto_transfer_noop"

    if operation == "CREATE":
        options = arguments.get("options") or {}
        auto_transfer_id = str(options["auto_transfer_id"])
        record = {
            "auto_transfer_id": auto_transfer_id,
            "source_account_id": arguments["source_account_id"],
            "target_id": arguments["target_id"],
            "status": "ACTIVE",
            "schedule": arguments["schedule"],
            "amount_krw": arguments.get("amount_krw"),
            "success_count": 0,
            "failed_count": 0,
        }
        _copy_option_fields(
            record,
            options,
            ("saving_range", "min_amount_krw", "max_amount_krw", "not_first_run_date"),
        )
        db.auto_transfer_rules.data[auto_transfer_id] = record
        return "applied_auto_transfer_config"

    raise UnsupportedMutatingActionError(
        f"configure_auto_transfer operation {operation!r} is not implemented"
    )


def _replay_request_interest_payment(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    target_id = str(arguments["target_id"])
    _, target = _find_record_by_id(db, target_id)

    interest_amount = _numeric_value(options["interest_amount_krw"])
    _credit_record_if_balance_backed(target, interest_amount)
    if "accrued_interest_krw" in target:
        target["accrued_interest_krw"] = 0
    target["last_interest_paid_at"] = _tool_timestamp()
    target["last_interest_payment_reason"] = options.get("reason")
    return "applied_interest_payment"


def _dispute_id(arguments: dict[str, Any]) -> str:
    target_id = str(arguments["target_id"])
    reason = str(arguments["reason"]).lower()
    return f"dispute_{target_id}_{reason}"


def _dispute_status(options: dict[str, Any]) -> str:
    if "investigation_status" in options:
        return str(options["investigation_status"])
    if options.get("compensation_approved") is False:
        return "REJECTED"
    if options.get("reactivation_approved") is False:
        return "LOGGED_REACTIVATION_REJECTED"
    return "LOGGED"


def _issue_new_card(db: KakaoBankDB, arguments: dict[str, Any]) -> None:
    card_id = str(arguments["card_id"])
    wallet_id = str(arguments["wallet_id"])
    db.cards.data[card_id] = {
        "card_id": card_id,
        "customer_id": arguments["customer_id"],
        "linked_account_or_wallet_id": wallet_id,
        "wallet_id": wallet_id,
        "product_name": "mini카드",
        "status": str(arguments.get("new_status", "ACTIVE")),
        "issued_at": _tool_timestamp(),
        "daily_limit_krw": 500000,
        "monthly_limit_krw": 2000000,
    }
    if "expected_valid_until" in arguments:
        db.cards.data[card_id]["expires_at"] = arguments["expected_valid_until"]

    issue_fee = _numeric_value(arguments.get("issue_fee_krw"))
    if issue_fee:
        _, wallet = _find_record_by_id(db, wallet_id)
        _debit_record_if_balance_backed(wallet, issue_fee)


def _reissue_card(db: KakaoBankDB, arguments: dict[str, Any]) -> None:
    card_id = str(arguments["card_id"])
    _, card = _find_record_by_id(db, card_id)
    if "expected_valid_until" in arguments:
        valid_until = arguments["expected_valid_until"]
        if "valid_until" in card:
            card["valid_until"] = valid_until
        else:
            card["expires_at"] = valid_until


def _reissue_card_without_tmoney_transfer(
    db: KakaoBankDB, arguments: dict[str, Any]
) -> None:
    _, old_card = _find_record_by_id(db, str(arguments["card_id"]))
    new_card_id = str(arguments["new_card_id"])
    new_card = deepcopy(old_card)
    new_card["card_id"] = new_card_id
    new_card["status"] = "ACTIVE"
    new_card["lost_reported_at"] = None
    new_card["physical_card_possession"] = True
    new_card["t_money_balance_krw"] = arguments.get("new_card_t_money_balance_krw", 0)
    if "expected_valid_until" in arguments:
        new_card["expires_at"] = arguments["expected_valid_until"]
    db.cards.data[new_card_id] = new_card


def _replay_update_loan_contract_state(db: KakaoBankDB, action: dict[str, Any]) -> str:
    arguments = action.get("arguments") or {}
    options = arguments.get("options") or {}
    operation = str(arguments["operation"])
    loan_id = str(arguments["loan_id"])
    reason = str(arguments.get("reason", ""))
    effective_at = _tool_timestamp()

    if operation == "REJECT_LEASE_CONTRACT_REVISION":
        return "rejected_loan_update_noop"

    if operation == "MARK_EXECUTION_BLOCKED":
        _update_application_decision(db, arguments, "EXECUTION_BLOCKED")
        return "applied_loan_contract_state"

    _, loan = _find_record_by_id(db, loan_id)
    if operation == "STOP_EXECUTION_BEFORE_DISBURSEMENT":
        loan["status"] = "EXECUTION_STOPPED"
        loan["execution_block_reason"] = reason
        loan["execution_stopped_at"] = effective_at
        return "applied_loan_contract_state"

    if operation in {
        "WITHDRAW_CONTRACT_WITHIN_COOLING_OFF",
        "PROCESS_LOAN_WITHDRAWAL_RIGHT",
    }:
        loan["status"] = "WITHDRAWN"
        loan["accelerated"] = False
        loan["acceleration_reason"] = None
        _set_required_document_status(db, loan_id, "LOAN_WITHDRAWAL_NOTICE", "ACCEPTED")
        if operation == "PROCESS_LOAN_WITHDRAWAL_RIGHT":
            _set_related_record_status(
                db, "vehicle_purchase_cases", loan_id, "LOAN_WITHDRAWN_BY_CUSTOMER"
            )
        return "applied_loan_contract_state"

    if operation in {
        "EXECUTE_LEASE_LOAN_TO_LANDLORD",
        "RESUME_EXECUTION_WITH_UPDATED_LANDLORD_ACCOUNT",
        "EXECUTE_SGI_LEASE_REFINANCE_REMAINING_TO_LANDLORD",
    }:
        _execute_lease_loan(db, loan, arguments, options)
        return "applied_loan_contract_state"

    if operation == "START_COLLATERAL_ENFORCEMENT":
        loan["status"] = "COLLATERAL_ENFORCEMENT"
        loan["collateral_enforcement_reason"] = reason
        collateral_id = options.get("collateral_id")
        if collateral_id:
            _set_record_status(db, str(collateral_id), "COLLATERAL_EXECUTION_STARTED")
            _, collateral = _find_record_by_id(db, str(collateral_id))
            collateral["collateral_execution_started_at"] = effective_at
            _copy_option_fields(
                collateral,
                options,
                ("ownership_loss_risk_disclosed", "legal_procedure_required"),
            )
        return "applied_loan_contract_state"

    if operation == "MARK_PURCHASE_MORTGAGE_EXECUTION_NOT_RUN":
        loan["status"] = "EXECUTION_NOT_RUN"
        loan["execution_not_run_reason"] = reason
        return "applied_loan_contract_state"

    if operation == "VERIFY_USED_CAR_TITLE_TRANSFER":
        loan["status"] = "EXECUTED"
        loan["accelerated"] = False
        loan["acceleration_reason"] = None
        _set_related_record_status(
            db, "vehicle_purchase_cases", loan_id, "TITLE_TRANSFER_VERIFIED"
        )
        _set_required_document_status(
            db, loan_id, "VEHICLE_TITLE_TRANSFER_REGISTRY_CHECK", "ACCEPTED"
        )
        return "applied_loan_contract_state"

    if operation.startswith("ACCELERATE"):
        _accelerate_loan(db, loan, loan_id, reason, effective_at, options)
        return "applied_loan_contract_state"

    raise UnsupportedMutatingActionError(
        f"update_loan_contract_state operation {operation!r} is not implemented"
    )


def _execute_lease_loan(
    db: KakaoBankDB,
    loan: dict[str, Any],
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    loan["status"] = "EXECUTED"
    loan["executed_at"] = _tool_timestamp()
    amount = _numeric_value(
        options.get("landlord_disbursement_amount_krw")
        or options.get("disbursement_amount_krw")
        or options.get("approved_amount_krw")
    )
    loan["outstanding_krw"] = amount if amount else loan.get("outstanding_krw", 0)
    if "principal_balance" in loan:
        loan["principal_balance"] = loan.get("principal_krw", amount)

    _update_application_decision(db, arguments, "EXECUTED")
    required_document = options.get("required_document_to_create")
    if required_document:
        _upsert_required_document(
            db,
            target_id=str(arguments["loan_id"]),
            document_type=required_document["document_type"],
            status=required_document["status"],
            document_id=required_document["document_id"],
            due_date=required_document.get("deadline"),
        )

    old_proof_id = options.get("old_loan_repayment_proof_document_id")
    if old_proof_id:
        _upsert_required_document(
            db,
            target_id=str(arguments["loan_id"]),
            document_type="OLD_LOAN_FULL_REPAYMENT_PROOF",
            status="VERIFIED",
            document_id=str(old_proof_id),
        )


def _update_application_decision(
    db: KakaoBankDB, arguments: dict[str, Any], status: str
) -> None:
    application_id = arguments.get("application_id")
    if not application_id:
        return
    application = _find_record_by_id_optional(db, str(application_id))
    if application is None:
        return
    application[1]["status"] = status
    application[1]["decision_reason"] = arguments.get("reason")
    if "effective_at" in arguments:
        application[1]["decision_at"] = _tool_timestamp()


def _accelerate_loan(
    db: KakaoBankDB,
    loan: dict[str, Any],
    loan_id: str,
    reason: str,
    effective_at: Any,
    options: dict[str, Any],
) -> None:
    loan["status"] = "ACCELERATED"
    loan["accelerated"] = True
    loan["acceleration_reason"] = reason
    loan["accelerated_at"] = effective_at
    loan["immediate_repayment_required"] = options.get(
        "immediate_repayment_required", True
    )

    if "document_type" in options:
        _upsert_required_document(
            db,
            target_id=loan_id,
            document_type=options["document_type"],
            status=options.get("document_status", "OVERDUE"),
            document_id=options.get("document_id"),
        )
    else:
        _apply_reason_based_document_status(db, loan_id, reason)

    if "lease_id" in options:
        _set_lease_status_for_acceleration(db, str(options["lease_id"]), reason)
    if "document_id" in options and "lease_contracts" in db.table_names:
        lease_id = _find_related_lease_contract_id(db, loan_id)
        if lease_id:
            _set_lease_status_for_acceleration(db, lease_id, reason)

    vehicle_status = _vehicle_status_for_acceleration_reason(reason)
    if vehicle_status:
        _set_related_record_status(
            db, "vehicle_purchase_cases", loan_id, vehicle_status
        )


def _apply_reason_based_document_status(
    db: KakaoBankDB, loan_id: str, reason: str
) -> None:
    mapping = {
        "POST_USE_INSPECTION_STATEMENT_NOT_SUBMITTED_WITHIN_3_MONTHS": (
            "BUSINESS_LOAN_USE_OF_FUNDS_STATEMENT",
            "OVERDUE_SANCTION_APPLIED",
        ),
        "BUSINESS_CREDIT_FUNDS_USED_OUTSIDE_APPROVED_PURPOSE": (
            "BUSINESS_LOAN_USE_OF_FUNDS_REVIEW",
            "VIOLATION_CONFIRMED",
        ),
        "HIGH_CREDIT_REGULATED_AREA_HOME_PURCHASE_VIOLATION": (
            "HIGH_CREDIT_PURPOSE_MANAGEMENT_ADDENDUM",
            "ACCEPTED",
        ),
        "MORTGAGE_MOVE_IN_PROOF_NOT_SUBMITTED_WITHIN_6_MONTHS": (
            "MORTGAGE_MOVE_IN_PROOF",
            "OVERDUE",
        ),
        "UNDISCLOSED_HOUSEHOLD_PRESALE_RIGHT_AT_EXECUTION": (
            "PURCHASE_PURPOSE_HOME_NOTICE_ADDENDUM",
            "ACCEPTED",
        ),
        "USED_CAR_LOAN_FUNDS_USED_OUTSIDE_VEHICLE_PURCHASE": (
            "VEHICLE_TITLE_TRANSFER_REGISTRY_CHECK",
            "PENDING",
        ),
        "USED_CAR_SALE_CANCEL_AFTER_DEALER_PAYMENT": (
            "VEHICLE_TITLE_TRANSFER_REGISTRY_CHECK",
            "NOT_APPLICABLE_SALE_CANCELLED",
        ),
        "TITLE_TRANSFER_NOT_COMPLETED_WITHIN_15_DAYS": (
            "VEHICLE_TITLE_TRANSFER_REGISTRY_CHECK",
            "OVERDUE",
        ),
        "TS_REGISTRY_SHOWS_TITLE_TRANSFER_NOT_COMPLETED": (
            "VEHICLE_TITLE_TRANSFER_REGISTRY_CHECK",
            "REJECTED_TS_REGISTRY_NOT_TRANSFERRED",
        ),
    }
    document_update = mapping.get(reason)
    if document_update:
        _set_required_document_status(db, loan_id, *document_update)


def _vehicle_status_for_acceleration_reason(reason: str) -> str | None:
    return {
        "USED_CAR_LOAN_FUNDS_USED_OUTSIDE_VEHICLE_PURCHASE": "NON_VEHICLE_USE_REPAYMENT_REQUIRED",
        "USED_CAR_SALE_CANCEL_AFTER_DEALER_PAYMENT": "SALE_CANCELLED_REPAYMENT_REQUIRED",
        "TITLE_TRANSFER_NOT_COMPLETED_WITHIN_15_DAYS": "TITLE_TRANSFER_OVERDUE_REPAYMENT_REQUIRED",
        "TS_REGISTRY_SHOWS_TITLE_TRANSFER_NOT_COMPLETED": "TS_REGISTRY_NOT_TRANSFERRED_REPAYMENT_REQUIRED",
    }.get(reason)


def _set_lease_status_for_acceleration(
    db: KakaoBankDB, lease_id: str, reason: str
) -> None:
    status = {
        "SGI_SUBMITTED_DOCUMENT_FALSE": "FALSE_DOCUMENT_CONFIRMED",
        "SGI_OPPOSING_POWER_LOST_DURING_TERM": "OPPOSING_POWER_LOST",
        "SGI_OPPOSING_POWER_NOT_SECURED_WITHIN_3_BUSINESS_DAYS": "OPPOSING_POWER_NOT_SECURED_OVERDUE",
    }.get(reason)
    if status:
        _set_record_status(db, lease_id, status)


def _find_related_lease_contract_id(db: KakaoBankDB, loan_id: str) -> str | None:
    for lease_id, record in db.lease_contracts.data.items():
        if record.get("loan_id") == loan_id:
            return lease_id
    return None


def _set_related_record_status(
    db: KakaoBankDB, table_name: str, loan_id: str, status: str
) -> None:
    table = db.get_table(table_name)
    if table is None:
        return
    for record in table.data.values():
        if record.get("loan_id") == loan_id:
            record["status"] = status


def _get_or_create_remittance_case(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    remittance_id = str(options["remittance_id"])
    existing = db.remittance_cases.data.get(remittance_id)
    if existing is not None:
        return existing

    remittance = {
        "remittance_id": remittance_id,
        "customer_id": arguments["customer_id"],
        "direction": _base_remittance_direction(str(arguments["direction"])),
        "status": "PENDING",
        "amount": arguments["amount"],
        "currency": arguments["currency"],
        "country": arguments["country"],
        "purpose_code": arguments["purpose_code"],
    }
    _copy_remittance_options(
        remittance,
        options,
        (
            "source_account_id",
            "target_account_id",
            "recipient_name",
            "recipient_country",
            "recipient_relationship",
            "requested_sender_name",
        ),
    )
    db.remittance_cases.data[remittance_id] = remittance
    return remittance


GENERATED_REMITTANCE_TRANSACTION_PREFIXES: dict[str, dict[str, str]] = {
    "DOLLARBOX_GIFT_AUTO_CANCEL": {
        "refund_transaction_id": "txn_dollar_gift_refund",
    },
    "DOLLARBOX_GIFT_RECEIVE": {
        "receive_transaction_id": "txn_dollar_gift_receive",
    },
    "INBOUND_BULK_DEPOSIT": {
        "transaction_id": "txn_inbound_remit",
    },
    "INBOUND_IMMEDIATE_DEPOSIT": {
        "transaction_id": "txn_inbound_remit",
    },
    "INBOUND_RETURN_INFO_MISMATCH": {
        "return_transaction_id": "txn_inbound_return",
    },
    "OUTBOUND_BENEFICIARY_INFO_AUTO_CANCEL": {
        "transaction_id": "txn_outbound_auto_cancel_return",
    },
    "OUTBOUND_NO_DOCUMENT": {
        "transaction_id": "txn_outbound_remit",
    },
    "OUTBOUND_RETURN_SETTLEMENT": {
        "transaction_id": "txn_outbound_return_settlement",
    },
}


def _generated_remittance_transaction_id(
    direction: str,
    options: dict[str, Any],
    field_name: str,
) -> str | None:
    supplied = options.get(field_name)
    if supplied:
        return str(supplied)
    prefix = GENERATED_REMITTANCE_TRANSACTION_PREFIXES.get(direction, {}).get(
        field_name
    )
    if prefix is None:
        return None
    return f"{prefix}_{_record_numeric_suffix(str(options['remittance_id']))}"


def _record_numeric_suffix(record_id: str) -> str:
    suffix = record_id.rsplit("_", 1)[-1]
    if len(suffix) != 3 or not suffix.isdigit():
        raise ReplayError(
            f"cannot generate deterministic ID from record id {record_id!r}"
        )
    return suffix


def _base_remittance_direction(direction: str) -> str:
    if direction.startswith("OUTBOUND_"):
        return "OUTBOUND"
    if direction.startswith("INBOUND_"):
        return "INBOUND"
    if direction.startswith("DOLLARBOX_GIFT_"):
        return "DOLLARBOX_GIFT_SEND"
    return direction


def _copy_remittance_options(
    remittance: dict[str, Any],
    options: dict[str, Any],
    field_names: tuple[str, ...],
) -> None:
    for field_name in field_names:
        if field_name in options:
            remittance[field_name] = options[field_name]


def _remove_pending_id(
    record: dict[str, Any],
    field_name: str,
    pending_id: str,
) -> None:
    pending_ids = record.get(field_name)
    if not isinstance(pending_ids, list):
        return
    record[field_name] = [item for item in pending_ids if item != pending_id]


def _upsert_transaction(
    db: KakaoBankDB,
    *,
    transaction_id: str | None,
    record_id: str,
    amount: int | float,
    currency: str,
    transaction_type: str,
) -> None:
    if not transaction_id:
        return

    transaction: dict[str, Any] = {
        "transaction_id": transaction_id,
        "transaction_type": transaction_type,
        "status": "POSTED",
        "currency": currency,
    }
    if record_id in db.accounts.data:
        transaction["account_id"] = record_id
    elif record_id in db.savings_boxes.data:
        transaction["box_id"] = record_id
    else:
        transaction["record_id"] = record_id

    if currency == "KRW":
        transaction["amount_krw"] = amount
    else:
        transaction["amount"] = amount

    db.transactions.data[transaction_id] = transaction


def _increment_remittance_profile_amount(
    db: KakaoBankDB,
    *,
    customer_id: str,
    field_name: str,
    amount: int | float,
) -> None:
    profile = _find_remittance_profile_for_customer(db, customer_id)
    if profile is None:
        return
    profile[field_name] = _numeric_value(profile.get(field_name)) + amount


def _set_remittance_profile_amount(
    db: KakaoBankDB,
    *,
    customer_id: str,
    field_name: str,
    amount: Any,
) -> None:
    if amount is None:
        return
    profile = _find_remittance_profile_for_customer(db, customer_id)
    if profile is None:
        return
    profile[field_name] = amount


def _find_remittance_profile_for_customer(
    db: KakaoBankDB, customer_id: str
) -> dict[str, Any] | None:
    for profile in db.remittance_profiles.data.values():
        if profile.get("customer_id") == customer_id:
            return profile
    return None


def _upsert_card_order(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    *,
    status: str,
) -> None:
    order_id = arguments.get("order_id")
    if not order_id:
        return

    card_order_id = str(order_id)
    order = {
        "card_order_id": card_order_id,
        "order_id": card_order_id,
        "card_id": arguments.get("card_id"),
        "customer_id": arguments.get("customer_id"),
        "wallet_id": arguments.get("wallet_id"),
        "status": status,
        "reason": arguments.get("reason"),
        "requested_at": _tool_timestamp(),
        "new_card_created": arguments.get("new_card_created", status == "APPROVED"),
    }
    _copy_option_fields(
        order,
        arguments,
        (
            "existing_card_id",
            "new_card_id",
            "expected_valid_until",
            "guardian_consent_valid",
            "customer_age",
            "t_money_balance_transferred",
            "t_money_refund_processed",
            "old_t_money_balance_krw",
            "new_card_t_money_balance_krw",
        ),
    )
    db.card_orders.data[card_order_id] = order


def _close_refinanced_old_loan(db: KakaoBankDB, refinance: dict[str, Any]) -> None:
    old_loan_id = refinance.get("old_loan_id")
    if not old_loan_id:
        return
    old_loan = _find_record_by_id_optional(db, str(old_loan_id))
    if old_loan is None:
        return
    old_loan[1]["status"] = "CLOSED"
    if "outstanding_krw" in old_loan[1]:
        old_loan[1]["outstanding_krw"] = 0


def _set_refinance_new_loan_status(
    db: KakaoBankDB, refinance: dict[str, Any], status: str
) -> None:
    new_loan_id = refinance.get("new_loan_id")
    if not new_loan_id:
        return
    new_loan = _find_record_by_id_optional(db, str(new_loan_id))
    if new_loan is None:
        return
    new_loan[1]["status"] = status
    if status == "ACTIVE":
        new_loan[1]["outstanding_krw"] = new_loan[1].get("principal_krw", 0)
        if isinstance(new_loan[1].get("restriction_flags"), list):
            new_loan[1]["restriction_flags"] = [
                flag
                for flag in new_loan[1]["restriction_flags"]
                if flag
                not in {
                    "PENDING_REFINANCE_EXECUTION",
                    "REFINANCE_PENDING_OLD_REPAYMENT",
                }
            ]


def _request_lien_release(db: KakaoBankDB, refinance: dict[str, Any]) -> None:
    old_loan_id = refinance.get("old_loan_id")
    if not old_loan_id:
        return
    for collateral in db.mortgage_collateral.data.values():
        if collateral.get("old_loan_id") == old_loan_id:
            collateral["status"] = "LIEN_RELEASE_REQUESTED"
            collateral["existing_lien_release_requested"] = True
            return


def _set_required_document_status(
    db: KakaoBankDB,
    target_id: str,
    document_type: str,
    status: str,
) -> None:
    for record in db.required_documents.data.values():
        if (
            record.get("target_id") == target_id
            and record.get("document_type") == document_type
        ):
            record["status"] = status
            return
    _upsert_required_document(
        db,
        target_id=target_id,
        document_type=document_type,
        status=status,
    )


def _upsert_required_document(
    db: KakaoBankDB,
    *,
    target_id: str,
    document_type: str,
    status: str,
    document_id: str | None = None,
    due_date: str | None = None,
) -> None:
    if document_id is None:
        document_id = f"required_{target_id}_{document_type}".lower()
    db.required_documents.data[document_id] = {
        "document_id": document_id,
        "target_id": target_id,
        "target_type": "loan",
        "document_type": document_type,
        "status": status,
        "due_date": due_date,
    }


def _is_rejected_transfer(arguments: dict[str, Any], transfer_type: str) -> bool:
    if "REJECT" in transfer_type:
        return True
    if arguments.get("executed") is False:
        return True
    if arguments.get("payment_approved") is False:
        return True
    return False


def _apply_transfer_record_updates(
    db: KakaoBankDB,
    arguments: dict[str, Any],
    *,
    source: tuple[str, dict[str, Any]] | None,
    target: tuple[str, dict[str, Any]] | None,
) -> None:
    _apply_transaction_record_updates(db, arguments)

    if source is not None:
        _apply_source_record_updates(source[1], arguments)
    if target is not None:
        _apply_target_record_updates(target[1], arguments)


def _apply_transaction_record_updates(
    db: KakaoBankDB, arguments: dict[str, Any]
) -> None:
    source_id = arguments.get("source_id")
    if not source_id:
        return

    source = _find_record_by_id_optional(db, str(source_id))
    if source is None or source[0] != "transactions":
        return

    record = source[1]
    if "new_transfer_status" in arguments:
        record["status"] = arguments["new_transfer_status"]
    if "new_transaction_status" in arguments:
        record["status"] = arguments["new_transaction_status"]
    if "refunded_at" in arguments:
        record["refunded_at"] = _tool_timestamp()


def _apply_source_record_updates(
    source_record: dict[str, Any],
    arguments: dict[str, Any],
) -> None:
    if "remaining_principal_krw" in arguments and "principal_krw" in source_record:
        source_record["principal_krw"] = arguments["remaining_principal_krw"]

    if "emergency_withdrawal_count_after" in arguments:
        _set_first_existing_field(
            source_record,
            (
                "emergency_withdrawal_count_current_term",
                "emergency_withdrawal_count",
            ),
            arguments["emergency_withdrawal_count_after"],
        )


def _apply_target_record_updates(
    target_record: dict[str, Any],
    arguments: dict[str, Any],
) -> None:
    update_map = {
        "new_current_month_contribution_krw": "current_month_contribution_krw",
        "new_today_payment_count": "today_payment_count",
        "new_cumulative_payment_count": "cumulative_payment_count",
        "new_daily_preferential_rate_pp": "daily_preferential_rate_pp",
        "new_bonus_preferential_rate_pp": "bonus_preferential_rate_pp",
    }
    for argument_name, field_name in update_map.items():
        if argument_name in arguments:
            target_record[field_name] = arguments[argument_name]

    if "payment_date" in arguments:
        target_record["last_payment_date"] = arguments["payment_date"]


def _set_first_existing_field(
    record: dict[str, Any],
    field_names: tuple[str, ...],
    value: Any,
) -> None:
    for field_name in field_names:
        if field_name in record:
            record[field_name] = value
            return


def _find_record_by_id(db: KakaoBankDB, record_id: str) -> tuple[str, dict[str, Any]]:
    for table_name, table in db.iter_tables():
        record = table.data.get(record_id)
        if record is not None:
            return table_name, record
    raise ReplayTargetRecordNotFoundError(f"record not found for replay: {record_id}")


def _find_record_by_id_optional(
    db: KakaoBankDB, record_id: str
) -> tuple[str, dict[str, Any]] | None:
    for table_name, table in db.iter_tables():
        record = table.data.get(record_id)
        if record is not None:
            return table_name, record
    return None


def _set_record_status(db: KakaoBankDB, record_id: str, status: str) -> None:
    _, record = _find_record_by_id(db, record_id)
    record["status"] = status


def _transfer_box_balance_to_base_account(
    db: KakaoBankDB,
    source_record: dict[str, Any],
    options: dict[str, Any],
) -> None:
    transfer_amount = _numeric_value(options.get("transfer_amount_krw"))
    base_account_id = str(
        options.get("base_account_id") or source_record["base_account_id"]
    )
    _transfer_krw_balance(
        db,
        source_record=source_record,
        target_id=base_account_id,
        amount=transfer_amount,
        source_balance_field="balance",
    )
    if "accrued_interest_krw" in source_record:
        source_record["accrued_interest_krw"] = 0


def _close_record_book_section(db: KakaoBankDB, section_record: dict[str, Any]) -> None:
    parent_account_id = str(section_record["parent_id"])
    _, parent_account = _find_record_by_id(db, parent_account_id)
    linked_account_id = str(parent_account["linked_account_id"])
    amount = _numeric_value(section_record.get("balance_krw"))

    _transfer_krw_balance(
        db,
        source_record=section_record,
        target_id=linked_account_id,
        amount=amount,
        source_balance_field="balance_krw",
    )
    parent_account["status"] = "CLOSED"
    parent_account["balance_krw"] = 0
    if "non_interest_section_count" in parent_account:
        parent_account["non_interest_section_count"] = 0


def _convert_group_account_service(
    db: KakaoBankDB,
    service_record: dict[str, Any],
    options: dict[str, Any],
) -> None:
    linked_account_id = service_record.get("linked_account_id")
    if not linked_account_id:
        return

    _, account = _find_record_by_id(db, str(linked_account_id))
    if options.get("convert_target_account_to") == "GENERAL_DEMAND_DEPOSIT":
        account["product_name"] = "입출금통장"

    release_flags = set(options.get("release_restriction_flags") or [])
    if release_flags and isinstance(account.get("restriction_flags"), list):
        account["restriction_flags"] = [
            flag for flag in account["restriction_flags"] if flag not in release_flags
        ]
    if "linked_service_ids" in account and isinstance(
        account["linked_service_ids"], list
    ):
        account["linked_service_ids"] = [
            service_id
            for service_id in account["linked_service_ids"]
            if service_id != service_record.get("service_id")
        ]


def _transfer_krw_balance(
    db: KakaoBankDB,
    *,
    source_record: dict[str, Any],
    target_id: str,
    amount: int | float,
    source_balance_field: str,
) -> None:
    _, target_record = _find_record_by_id(db, target_id)
    target_balance_field = (
        "balance_krw" if "balance_krw" in target_record else "balance"
    )
    target_record[target_balance_field] = (
        _numeric_value(target_record.get(target_balance_field)) + amount
    )
    source_record[source_balance_field] = 0


def _debit_record_if_balance_backed(
    record: dict[str, Any], amount: int | float
) -> None:
    balance_field = _balance_field(record)
    if balance_field is None:
        return
    if amount < 0:
        raise ReplayError(f"cannot debit negative amount: {amount!r}")
    balance = _numeric_value(record.get(balance_field))
    if balance < amount:
        raise ReplayError(
            f"insufficient balance for debit: available={balance!r}, amount={amount!r}"
        )
    record[balance_field] = balance - amount


def _credit_record_if_balance_backed(
    record: dict[str, Any], amount: int | float
) -> None:
    balance_field = _balance_field(record)
    if balance_field is None:
        return
    record[balance_field] = _numeric_value(record.get(balance_field)) + amount


def _balance_field(record: dict[str, Any]) -> str | None:
    for field_name in ("balance_krw", "principal_krw", "balance"):
        if field_name in record:
            return field_name
    return None


def _numeric_value(value: Any) -> int | float:
    if isinstance(value, int | float):
        return value
    if value is None:
        return 0
    raise ReplayError(f"expected numeric value during replay: {value!r}")


def _first_numeric_option(
    options: dict[str, Any], field_names: tuple[str, ...]
) -> int | float:
    for field_name in field_names:
        if field_name in options:
            return _numeric_value(options[field_name])
    raise ReplayError(f"expected one of numeric options {field_names!r}")


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)
