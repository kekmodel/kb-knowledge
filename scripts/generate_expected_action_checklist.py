from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kb_knowledge.kakaobank.replay import build_empty_domain_db, replay_expected_actions  # noqa: E402
from kb_knowledge.kakaobank.runner import build_openai_tool_definitions  # noqa: E402
from kb_knowledge.kakaobank.tools import KakaoBankReadTools  # noqa: E402


TASKS_PATH = REPO_ROOT / "data/kakaobank_knowledge/v0/tasks/tasks.json"
DOCS_DIR = REPO_ROOT / "data/kakaobank_knowledge/v0/knowledge_base/documents"
DEFAULT_OUTPUT = REPO_ROOT / "docs/expected_action_manual_checklist.md"

READ_ACTION_NAMES = {
    "KB_search",
    "grep",
    "get_customer_profile",
    "get_account_or_contract",
}
CODE_FIELD_NAMES = {
    "operation",
    "transaction_type",
    "transfer_type",
    "close_type",
    "reason",
    "direction",
    "purpose_code",
    "expected_status",
    "expected_reason",
    "new_status",
    "new_card_status",
    "new_transfer_status",
    "original_transfer_status",
    "new_transaction_status",
    "rejection_reason_for_other_sources",
    "interest_rate_type_for_withdrawn_amount",
}
CODE_LITERAL_RE = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)+$")
ID_LITERAL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the manual expected-action audit checklist."
    )
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing checklist even if it contains manual audit results.",
    )
    args = parser.parse_args()

    if (
        args.output.exists()
        and not args.force
        and "## Manual Audit Summary" in args.output.read_text(encoding="utf-8")
    ):
        raise SystemExit(
            f"{args.output} already contains manual audit results; "
            "use --force only if you intentionally want to regenerate it."
        )

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    doc_ids = load_document_ids(args.docs_dir)
    tool_surface = json.dumps(
        build_openai_tool_definitions(retrieval_config="bm25_grep"),
        ensure_ascii=False,
        sort_keys=True,
    )
    read_tools = KakaoBankReadTools(build_empty_domain_db(), documents_dir=args.docs_dir)

    rows = [
        analyze_task(
            task,
            doc_ids=doc_ids,
            docs_dir=args.docs_dir,
            tool_surface=tool_surface,
            read_tools=read_tools,
        )
        for task in tasks
    ]

    output = render_markdown(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(args.output)


def load_document_ids(docs_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in docs_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        ids.add(str(data["id"]))
    return ids


def analyze_task(
    task: dict[str, Any],
    *,
    doc_ids: set[str],
    docs_dir: Path,
    tool_surface: str,
    read_tools: KakaoBankReadTools,
) -> dict[str, Any]:
    actions = task["evaluation_criteria"].get("actions", [])
    write_actions = [action for action in actions if action.get("name") not in READ_ACTION_NAMES]
    required_documents = [str(item) for item in task.get("required_documents") or []]
    missing_documents = [doc_id for doc_id in required_documents if doc_id not in doc_ids]
    query = expected_kb_query(actions)
    top10_ids = [
        str(document["id"])
        for document in read_tools.KB_search(query, top_k=10).get("documents", [])
    ] if query else []
    retrieval_top10_missing = [
        doc_id for doc_id in required_documents if doc_id not in top10_ids
    ]
    replay_ok, replay_error = replay_gold(task)
    hidden_code_literals = find_hidden_code_literals(
        task,
        required_documents=required_documents,
        docs_dir=docs_dir,
        tool_surface=tool_surface,
    )
    possible_unobservable_refs = find_possible_unobservable_refs(task)

    flags: list[str] = []
    if not replay_ok:
        flags.append(f"gold_replay_failed: {replay_error}")
    if missing_documents:
        flags.append("required_document_missing")
    if retrieval_top10_missing:
        flags.append("required_document_not_in_expected_query_top10")
    if len(write_actions) > 1:
        flags.append("multi_write_expected")
    if possible_unobservable_refs:
        flags.append("possible_unobservable_existing_ref")
    if hidden_code_literals:
        flags.append("code_literals_not_static_exposed")

    return {
        "task_id": task["id"],
        "user_prompt": task.get("user_prompt", ""),
        "required_documents": required_documents,
        "missing_documents": missing_documents,
        "expected_kb_query": query,
        "top10_ids": top10_ids,
        "retrieval_top10_missing": retrieval_top10_missing,
        "action_sequence": [str(action.get("name", "")) for action in actions],
        "write_summaries": [summarize_write_action(action) for action in write_actions],
        "write_count": len(write_actions),
        "flags": flags,
        "replay_ok": replay_ok,
        "replay_error": replay_error,
        "possible_unobservable_refs": possible_unobservable_refs,
        "hidden_code_literals": hidden_code_literals,
    }


def expected_kb_query(actions: list[dict[str, Any]]) -> str:
    for action in actions:
        if action.get("name") == "KB_search":
            return str((action.get("arguments") or {}).get("query") or "")
    return ""


def replay_gold(task: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        replay_expected_actions(task)
    except Exception as exc:  # noqa: BLE001 - audit output should keep going.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def summarize_write_action(action: dict[str, Any]) -> str:
    name = str(action.get("name") or "")
    args = action.get("arguments") or {}
    key_order = [
        "customer_id",
        "source_id",
        "source_account_id",
        "target_id",
        "destination_account_id",
        "amount",
        "currency",
        "operation",
        "transaction_type",
        "transfer_type",
        "close_type",
        "direction",
        "purpose_code",
        "expected_status",
        "expected_reason",
        "reason",
        "loan_id",
        "application_id",
        "refinance_id",
        "card_id",
        "wallet_id",
    ]
    option_key_order = [
        "operation",
        "close_type",
        "destination_account_id",
        "expected_status",
        "expected_reason",
        "status",
        "reason",
        "new_status",
        "new_card_status",
        "new_transfer_status",
    ]
    parts: list[str] = []
    for key in key_order:
        if key in args and args[key] is not None:
            parts.append(f"{key}={format_summary_value(args[key])}")
    options = args.get("options")
    if isinstance(options, dict):
        for key in option_key_order:
            if key in options and options[key] is not None:
                parts.append(f"options.{key}={format_summary_value(options[key])}")
    if len(parts) > 14:
        parts = parts[:14] + ["..."]
    return f"{name}(" + ", ".join(parts) + ")"


def format_summary_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def find_hidden_code_literals(
    task: dict[str, Any],
    *,
    required_documents: list[str],
    docs_dir: Path,
    tool_surface: str,
) -> list[str]:
    visible_text = tool_surface + "\n" + str(task.get("user_prompt") or "")
    for doc_id in required_documents:
        path = docs_dir / f"{doc_id}.json"
        if path.exists():
            visible_text += "\n" + path.read_text(encoding="utf-8")

    hidden: set[str] = set()
    for action in task["evaluation_criteria"].get("actions", []):
        if action.get("name") in READ_ACTION_NAMES:
            continue
        for path, value in iter_leaf_items(action.get("arguments") or {}):
            key = path_key(path)
            if not isinstance(value, str):
                continue
            if key not in CODE_FIELD_NAMES and not CODE_LITERAL_RE.match(value):
                continue
            if CODE_LITERAL_RE.match(value) and value not in visible_text:
                hidden.add(value)
    return sorted(hidden)


def find_possible_unobservable_refs(task: dict[str, Any]) -> list[str]:
    record_ids, field_ids = initial_record_and_field_ids(task)
    available = set(record_ids) | set(field_ids)
    generated: set[str] = set()
    refs: set[str] = set()

    for index, action in enumerate(task["evaluation_criteria"].get("actions", [])):
        name = str(action.get("name") or "")
        for path, value in existing_reference_values(action):
            if value not in available and value not in generated:
                refs.add(f"a{index}:{name}:{path}={value}")

        if name not in READ_ACTION_NAMES:
            generated.update(find_id_literals(action.get("arguments") or {}))

    return sorted(refs)


def existing_reference_values(action: dict[str, Any]) -> list[tuple[str, str]]:
    name = str(action.get("name") or "")
    args = action.get("arguments") or {}
    operation = str(args.get("operation") or (args.get("options") or {}).get("operation") or "")
    refs: list[tuple[str, str]] = []

    top_level_by_action = {
        "get_customer_profile": {"customer_id"},
        "get_account_or_contract": {"record_id"},
        "close_account_or_service": {
            "customer_id",
            "target_id",
            "destination_account_id",
            "also_close_service_id",
        },
        "execute_deposit_or_box_transfer": {
            "source_id",
            "source_account_id",
            "target_id",
            "requested_by_customer_id",
        },
        "open_or_enroll_product": {"customer_id", "source_account_id"},
        "request_maturity_or_extension": {"target_id"},
        "request_interest_payment": {"target_id"},
        "update_loan_contract_state": {"loan_id"},
        "process_refinance_request": {"refinance_id"},
        "create_loan_application": {"customer_id"},
        "configure_auto_transfer": {
            "source_account_id",
            "target_id",
            "existing_auto_transfer_id",
        },
        "execute_remittance_case": {"customer_id"},
        "update_card_state": {"customer_id", "wallet_id", "existing_card_id"},
        "file_dispute_or_objection": {"customer_id", "target_id"},
    }
    nested_by_action = {
        "close_account_or_service": {
            "destination_account_id",
            "source_account_id",
            "also_close_service_id",
            "service_id",
            "membership_id",
        },
        "execute_deposit_or_box_transfer": {
            "source_id",
            "source_account_id",
            "target_id",
            "destination_account_id",
            "wallet_id",
            "account_id",
            "contract_id",
        },
        "request_maturity_or_extension": {
            "destination_account_id",
            "source_account_id",
            "account_id",
            "contract_id",
        },
        "request_interest_payment": {"destination_id", "destination_account_id"},
        "update_loan_contract_state": {
            "loan_id",
            "application_id",
            "lease_contract_id",
            "vehicle_purchase_case_id",
            "collateral_id",
            "required_document_id",
            "account_id",
        },
        "process_refinance_request": {"old_loan_id", "new_loan_id", "collateral_id"},
        "create_loan_application": {"source_account_id", "collateral_id"},
        "configure_auto_transfer": {"source_account_id", "target_id"},
        "execute_remittance_case": {
            "source_account_id",
            "target_account_id",
            "target_id",
            "source_id",
            "wallet_id",
            "box_id",
            "dollarbox_id",
            "linked_account_id",
        },
        "update_card_state": {"wallet_id", "existing_card_id", "transaction_id"},
    }

    if name == "update_card_state" and operation not in {"ISSUE_NEW_CARD", "REJECT_NEW_ISSUE"}:
        top_level_by_action[name].add("card_id")
    if name == "configure_auto_transfer" and operation != "CREATE":
        nested_by_action[name].add("auto_transfer_id")

    for key in top_level_by_action.get(name, set()):
        value = args.get(key)
        if isinstance(value, str) and ID_LITERAL_RE.match(value):
            refs.append((key, value))

    nested_keys = nested_by_action.get(name, set())
    for path, value in iter_leaf_items(args):
        key = path_key(path)
        if key in nested_keys and isinstance(value, str) and ID_LITERAL_RE.match(value):
            refs.append((path, value))

    return sorted(set(refs))


def initial_record_and_field_ids(task: dict[str, Any]) -> tuple[set[str], set[str]]:
    agent_data = (
        ((task.get("initial_state") or {}).get("initialization_data") or {})
        .get("agent_data")
        or {}
    )
    record_ids: set[str] = set()
    field_ids: set[str] = set()
    for table in agent_data.values():
        data = table.get("data") if isinstance(table, dict) else None
        if not isinstance(data, dict):
            continue
        record_ids.update(str(record_id) for record_id in data)
        for record in data.values():
            field_ids.update(find_id_literals(record))
    return record_ids, field_ids


def find_id_literals(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            ids.update(find_id_literals(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(find_id_literals(item))
    elif isinstance(value, str) and ID_LITERAL_RE.match(value):
        ids.add(value)
    return ids


def iter_leaf_items(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaf_items(item, next_prefix)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_leaf_items(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def path_key(path: str) -> str:
    key = path.split(".")[-1]
    return re.sub(r"\[\d+\]", "", key)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    total_actions = sum(len(row["action_sequence"]) for row in rows)
    total_writes = sum(row["write_count"] for row in rows)
    replay_failures = [row for row in rows if not row["replay_ok"]]
    retrieval_misses = [row for row in rows if row["retrieval_top10_missing"]]
    missing_docs = [row for row in rows if row["missing_documents"]]
    unobservable = [row for row in rows if row["possible_unobservable_refs"]]
    multi_write = [row for row in rows if row["write_count"] > 1]
    hidden_codes = [row for row in rows if row["hidden_code_literals"]]

    lines: list[str] = [
        "# Expected Action Manual Audit Checklist",
        "",
        "This checklist is for a full manual audit of every v0 KakaoBank task's expected actions.",
        "The goal is to distinguish correct gold actions from task bugs, fixture visibility gaps, KB/query gaps, and tool-schema gaps.",
        "",
        "## Audit Rules",
        "",
        "- Do not treat successful gold replay as proof that the expected action is correct.",
        "- Because the benchmark is primarily DB-hash based, the audit target is the necessary final DB delta and the write tools that create it.",
        "- Read/search actions are important for fairness and trace quality, but they do not by themselves prove DB-hash correctness.",
        "- For every write action, verify policy support in `required_documents`, runtime-state support in `initial_state`, and deterministic replay semantics in `src/kb_knowledge/kakaobank/replay.py`.",
        "- Prefer checking the minimal sufficient write tool set first, then exact write arguments, then supporting read/search actions.",
        "- Any ID required in an expected action must be visible in runtime context, discoverable from a read record, or explicitly generated by an earlier write action.",
        "- Any exact code-like value required by DB-hash evaluation should be reasonably available to the assistant through tool schema, policy text, runtime state, or a deterministic convention.",
        "- Mark a task as a fixture/tooling problem when the business logic is right but the assistant cannot fairly observe the required ID or code.",
        "",
        "## Preflight Summary",
        "",
        f"- Tasks: {len(rows)}",
        f"- Expected actions: {total_actions}",
        f"- Expected write actions: {total_writes}",
        f"- Gold replay failures: {len(replay_failures)}",
        f"- Missing required document files: {len(missing_docs)}",
        f"- Required docs not in BM25 top-10 for expected `KB_search` query: {len(retrieval_misses)}",
        f"- Multi-write expected tasks: {len(multi_write)}",
        f"- Possible unobservable existing references: {len(unobservable)}",
        f"- Tasks with code-like literals not statically exposed in tool schema/user prompt/required docs: {len(hidden_codes)}",
        "",
    ]

    lines.extend(render_summary_list("Gold Replay Failures", replay_failures, "replay_error"))
    lines.extend(render_summary_list("Required Document Retrieval Misses", retrieval_misses, "retrieval_top10_missing"))
    lines.extend(render_summary_list("Possible Unobservable References", unobservable, "possible_unobservable_refs"))
    lines.extend(render_summary_list("Multi-Write Tasks", multi_write, "action_sequence"))
    lines.extend(
        [
            "## Per-Task Checklist",
            "",
            "For each task, fill one final verdict:",
            "`OK`, `EXPECTED_ACTION_BUG`, `FIXTURE_VISIBILITY_GAP`, `KB_OR_QUERY_GAP`, `TOOL_SCHEMA_GAP`, or `NEEDS_DISCUSSION`.",
            "",
        ]
    )

    for index, row in enumerate(rows, start=1):
        lines.extend(render_task(row, index))

    return "\n".join(lines).rstrip() + "\n"


def render_summary_list(title: str, rows: list[dict[str, Any]], field: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["- None", ""])
        return lines

    for row in rows:
        value = row[field]
        if isinstance(value, list):
            value_text = "; ".join(str(item) for item in value)
        else:
            value_text = str(value)
        lines.append(f"- `{row['task_id']}`: {value_text}")
    lines.append("")
    return lines


def render_task(row: dict[str, Any], index: int) -> list[str]:
    flags = ", ".join(row["flags"]) if row["flags"] else "none"
    required_docs = ", ".join(f"`{doc_id}`" for doc_id in row["required_documents"]) or "none"
    sequence = " -> ".join(f"`{name}`" for name in row["action_sequence"])
    writes = "; ".join(f"`{item}`" for item in row["write_summaries"]) or "none"
    top10 = ", ".join(f"`{doc_id}`" for doc_id in row["top10_ids"][:5]) or "none"
    unobservable = row["possible_unobservable_refs"]
    hidden_codes = row["hidden_code_literals"]

    lines = [
        f"### {index}. `{row['task_id']}`",
        "",
        "- [ ] Final verdict: `OK` / `EXPECTED_ACTION_BUG` / `FIXTURE_VISIBILITY_GAP` / `KB_OR_QUERY_GAP` / `TOOL_SCHEMA_GAP` / `NEEDS_DISCUSSION`",
        f"- User prompt: {row['user_prompt']}",
        f"- Required documents: {required_docs}",
        f"- Expected KB query: `{row['expected_kb_query']}`",
        f"- BM25 top-5 for expected query: {top10}",
        f"- Expected action sequence: {sequence}",
        f"- Write action count: {row['write_count']}",
        f"- Expected write actions: {writes}",
        f"- Auto flags: {flags}",
    ]
    if row["retrieval_top10_missing"]:
        lines.append(
            "- Retrieval miss to inspect: "
            + ", ".join(f"`{doc_id}`" for doc_id in row["retrieval_top10_missing"])
        )
    if unobservable:
        lines.append("- Possible unobservable refs:")
        for item in unobservable:
            lines.append(f"  - `{item}`")
    if hidden_codes:
        sample = ", ".join(f"`{code}`" for code in hidden_codes[:12])
        suffix = f" (+{len(hidden_codes) - 12} more)" if len(hidden_codes) > 12 else ""
        lines.append(f"- Code literals to verify exposure: {sample}{suffix}")
    else:
        lines.append("- Code literals to verify exposure: none")

    lines.extend(
        [
            "- Manual checks:",
            "  - [ ] User prompt intent and adversarial request are captured correctly.",
            "  - [ ] Required policy document(s) support the expected decision.",
            "  - [ ] Initial state contains the facts required for the expected branch.",
            "  - [ ] The necessary final DB delta is clear and DB-hash-verifiable.",
            "  - [ ] The expected write tool set is minimal and sufficient for that DB delta.",
            "  - [ ] Every write tool name is the right business operation for the policy-backed decision.",
            "  - [ ] Every write argument matches policy, runtime state, and replay semantics.",
            "  - [ ] Expected read/search actions expose or justify the records and policy facts needed before each write.",
            "  - [ ] No expected argument requires a hidden ID, hidden enum, or impossible inference.",
            "  - [ ] DB-only final-state equality is sufficient for this task, or extra action constraints are intentionally needed.",
            "- Notes:",
            "  - ",
            "",
        ]
    )
    return lines


if __name__ == "__main__":
    main()
