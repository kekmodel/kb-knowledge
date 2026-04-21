from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kb_knowledge.kakaobank.replay import evaluate_candidate_actions
from kb_knowledge.kakaobank.runner import build_openai_tool_definitions
from kb_knowledge.kakaobank.tool_arg_models import (
    pydantic_tool_parameters,
    validate_pydantic_tool_arguments,
)


ROOT = Path(__file__).resolve().parents[1]


def test_remittance_schema_matches_openai_structured_outputs_subset() -> None:
    schema = pydantic_tool_parameters("execute_remittance_case")

    assert schema is not None
    assert schema["type"] == "object"
    assert "$defs" not in schema
    assert "anyOf" not in schema
    assert len(schema["properties"]["options"]["anyOf"]) == 13
    assert _strict_object_schema_issues(schema) == []


def test_all_gold_remittance_actions_validate_against_pydantic_models() -> None:
    tasks_dir = ROOT / "data/kakaobank_knowledge/v0/tasks/cases"
    issues: list[tuple[str, str]] = []
    action_count = 0

    for path in sorted(tasks_dir.glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        for action in task.get("evaluation_criteria", {}).get("actions") or []:
            if action.get("name") != "execute_remittance_case":
                continue
            action_count += 1
            error = validate_pydantic_tool_arguments(
                "execute_remittance_case",
                action.get("arguments") or {},
            )
            if error is not None:
                issues.append((str(task.get("id", path.stem)), error))

    assert action_count == 13
    assert issues == []


def test_remittance_validation_rejects_wrong_generated_id() -> None:
    arguments = _gold_remittance_arguments(
        "kb_manual_dollarbox_gift_auto_cancel_refunds_sender"
    )
    arguments["options"]["refund_transaction_id"] = "txn_inbound_remit_016_refund"

    error = validate_pydantic_tool_arguments("execute_remittance_case", arguments)

    assert error is not None
    assert "refund_transaction_id" in error
    assert "txn_dollar_gift_refund_016" in error


def test_remittance_replay_generates_null_transaction_id() -> None:
    task = _load_task("kb_manual_dollarbox_gift_auto_cancel_refunds_sender")
    actions = json.loads(json.dumps(task["evaluation_criteria"]["actions"]))
    for action in actions:
        if action.get("name") == "execute_remittance_case":
            action["arguments"]["options"]["refund_transaction_id"] = None

    result = evaluate_candidate_actions(task, actions)

    assert result.passed is True


def test_strict_tool_schema_flag_marks_only_pydantic_tools() -> None:
    default_tools = build_openai_tool_definitions()
    strict_tools = build_openai_tool_definitions(strict_tool_schemas=True)

    default_functions = {
        tool["function"]["name"]: tool["function"] for tool in default_tools
    }
    strict_functions = {
        tool["function"]["name"]: tool["function"] for tool in strict_tools
    }

    assert "strict" not in default_functions["execute_remittance_case"]
    assert strict_functions["execute_remittance_case"]["strict"] is True
    assert "strict" not in strict_functions["open_or_enroll_product"]
    assert "strict" not in strict_functions["KB_search"]


def _strict_object_schema_issues(schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                properties = node.get("properties") or {}
                required = set(node.get("required") or [])
                if required != set(properties):
                    issues.append(f"{path}: required keys do not match properties")
                if node.get("additionalProperties") is not False:
                    issues.append(f"{path}: additionalProperties is not false")
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(schema, "$")
    return issues


def _gold_remittance_arguments(task_id: str) -> dict[str, Any]:
    task = _load_task(task_id)
    for action in task.get("evaluation_criteria", {}).get("actions") or []:
        if action.get("name") == "execute_remittance_case":
            return json.loads(json.dumps(action["arguments"]))
    raise AssertionError(f"{task_id} has no execute_remittance_case action")


def _load_task(task_id: str) -> dict[str, Any]:
    path = ROOT / f"data/kakaobank_knowledge/v0/tasks/cases/{task_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
