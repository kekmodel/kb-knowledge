from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from kb_knowledge.kakaobank.replay import (
    KAKAOBANK_KNOWLEDGE_DB_PATH,
    KAKAOBANK_KNOWLEDGE_TASKS_DIR,
    ReplayError,
    evaluate_candidate_actions,
    load_exported_task,
    replay_expected_actions,
    write_empty_domain_db,
)
from kb_knowledge.kakaobank.runner import (
    DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
    DEFAULT_OPENAI_MODEL,
    run_task_with_openai_compatible,
)
from kb_knowledge.kakaobank.tools import (
    DEFAULT_RETRIEVAL_CONFIG,
    SUPPORTED_RETRIEVAL_CONFIGS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb-knowledge",
        description="KakaoBank knowledge v0 evaluator and OpenAI-compatible runner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime_db = subparsers.add_parser(
        "build-kakaobank-runtime-db",
        help="Write the empty KakaoBank runtime DB fixture",
    )
    runtime_db.add_argument(
        "--output",
        default=str(KAKAOBANK_KNOWLEDGE_DB_PATH),
        help="Path to output db.json",
    )

    replay_tasks = subparsers.add_parser(
        "replay-kakaobank-tasks",
        help="Replay exported KakaoBank v0 tasks with deterministic DB replay",
    )
    replay_tasks.add_argument(
        "--tasks-dir",
        default=str(KAKAOBANK_KNOWLEDGE_TASKS_DIR),
        help="Directory for one exported task JSON per file",
    )

    evaluate_actions = subparsers.add_parser(
        "evaluate-kakaobank-actions",
        help="Evaluate candidate assistant tool calls by DB final-state equality",
    )
    evaluate_actions.add_argument("--task-id", required=True)
    evaluate_actions.add_argument("--actions-json", required=True)
    evaluate_actions.add_argument(
        "--tasks-dir",
        default=str(KAKAOBANK_KNOWLEDGE_TASKS_DIR),
        help="Directory for one exported task JSON per file",
    )

    run_task = subparsers.add_parser(
        "run-kakaobank-task",
        help=(
            "Run one DB-delta v0 task with one OpenAI-compatible assistant "
            "endpoint and evaluate DB state"
        ),
    )
    _add_runner_arguments(run_task)
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument(
        "--tasks-dir",
        default=str(KAKAOBANK_KNOWLEDGE_TASKS_DIR),
        help="Directory for one exported task JSON per file",
    )
    run_task.add_argument(
        "--output-actions-json",
        default=None,
        help="Optional path to write captured assistant actions",
    )
    run_task.add_argument(
        "--output-trace-json",
        default=None,
        help=(
            "Optional path to write the integrated assistant trace, including "
            "requests, assistant messages, tool calls, tool results, and DB hashes"
        ),
    )

    run_tasks = subparsers.add_parser(
        "run-kakaobank-tasks",
        help=(
            "Run exported DB-delta v0 tasks with one OpenAI-compatible assistant "
            "endpoint and report DB pass/fail"
        ),
    )
    _add_runner_arguments(run_tasks)
    run_tasks.add_argument(
        "--tasks-dir",
        default=str(KAKAOBANK_KNOWLEDGE_TASKS_DIR),
        help="Directory for one exported task JSON per file",
    )
    run_tasks.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=None,
        help="Optional task ID to run; repeat to run a subset",
    )
    run_tasks.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tasks to run after sorting",
    )
    run_tasks.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write per-task run results",
    )
    run_tasks.add_argument(
        "--output-jsonl",
        default=None,
        help=(
            "Optional path to append one trace-bearing JSON result per completed "
            "task while the batch is still running"
        ),
    )

    return parser


def _add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="Assistant model name served by the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get(
            "OPENAI_COMPATIBLE_ENDPOINT",
            os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_ENDPOINT),
        ),
        help=(
            "OpenAI-compatible assistant endpoint. Accepts either a full "
            "/chat/completions URL or a /v1 base URL."
        ),
    )
    parser.add_argument(
        "--retrieval-config",
        choices=SUPPORTED_RETRIEVAL_CONFIGS,
        default=DEFAULT_RETRIEVAL_CONFIG,
        help="Offline retrieval tools exposed to the assistant",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Optional environment variable containing the endpoint API key",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="HTTP timeout per assistant request",
    )
    parser.add_argument(
        "--max-tool-steps",
        type=int,
        default=12,
        help="Maximum assistant tool-call rounds before evaluation",
    )


def _normalize_candidate_actions(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, dict):
        for key in ("actions", "tool_calls", "assistant_actions"):
            if key in raw:
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise SystemExit(
            "actions JSON must be a list or contain an actions/tool_calls key"
        )

    normalized: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SystemExit(f"action at index {index} must be an object")

        if "function" in item and "name" not in item:
            function = item["function"]
            if not isinstance(function, dict):
                raise SystemExit(f"tool call at index {index} has invalid function")
            name = function.get("name")
            if not isinstance(name, str):
                raise SystemExit(f"tool call at index {index} is missing function.name")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
            if not isinstance(arguments, dict):
                raise SystemExit(f"tool call at index {index} has non-object arguments")
            normalized.append(
                {
                    "requestor": "assistant",
                    "name": name,
                    "arguments": arguments,
                }
            )
            continue

        normalized.append(item)

    return normalized


def _load_candidate_actions(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_candidate_actions(raw)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-kakaobank-runtime-db":
        path = write_empty_domain_db(Path(args.output))
        print(f"db: {path}")
        return

    if args.command == "replay-kakaobank-tasks":
        _replay_tasks(Path(args.tasks_dir))
        return

    if args.command == "evaluate-kakaobank-actions":
        _evaluate_actions(
            task_id=args.task_id,
            actions_json=Path(args.actions_json),
            tasks_dir=Path(args.tasks_dir),
        )
        return

    if args.command == "run-kakaobank-task":
        _run_one_task(args)
        return

    if args.command == "run-kakaobank-tasks":
        _run_task_batch(args)
        return

    raise SystemExit(f"unknown command: {args.command}")


def _replay_tasks(tasks_dir: Path) -> None:
    passed = 0
    failures: list[tuple[str, str, str]] = []

    for task_path in sorted(tasks_dir.glob("*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        try:
            replay_expected_actions(task)
        except ReplayError as exc:
            failures.append(
                (
                    str(task.get("id", task_path.stem)),
                    type(exc).__name__,
                    str(exc),
                )
            )
        else:
            passed += 1

    print(f"tasks: {passed + len(failures)}")
    print(f"passed: {passed}")
    print(f"failed: {len(failures)}")
    for task_id, error_type, message in failures[:20]:
        print(f"failure: {task_id}: {error_type}: {message}")
    if failures:
        raise SystemExit(1)


def _evaluate_actions(
    *,
    task_id: str,
    actions_json: Path,
    tasks_dir: Path,
) -> None:
    task = load_exported_task(task_id, tasks_dir=tasks_dir)
    actions = _load_candidate_actions(actions_json)
    result = evaluate_candidate_actions(task, actions)

    print(f"task: {result.task_id}")
    print(f"passed: {str(result.passed).lower()}")
    print(f"initialized_hash: {result.initialized_hash}")
    print(f"expected_final_hash: {result.expected_final_hash}")
    print(f"actual_final_hash: {result.actual_final_hash}")
    if result.error is not None:
        print(f"error: {result.error}")
    if not result.passed:
        raise SystemExit(1)


def _run_one_task(args: argparse.Namespace) -> None:
    task = load_exported_task(args.task_id, tasks_dir=Path(args.tasks_dir))
    try:
        result = run_task_with_openai_compatible(
            task,
            model=args.model,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            retrieval_config=args.retrieval_config,
            max_tool_steps=args.max_tool_steps,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should surface endpoint failures.
        raise SystemExit(
            f"assistant endpoint run failed: {type(exc).__name__}: {exc}"
        ) from exc

    if args.output_actions_json:
        output_path = Path(args.output_actions_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.actions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_trace_json:
        output_path = Path(args.output_trace_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"task: {result.task_id}")
    print("task_set: kakaobank_knowledge v0 DB-delta only")
    print(f"model: {args.model}")
    print(f"endpoint: {args.endpoint}")
    print(f"retrieval_config: {args.retrieval_config}")
    print(f"passed: {str(result.passed).lower()}")
    print(f"stopped_reason: {result.stopped_reason}")
    print(f"actions: {len(result.actions)}")
    print(f"expected_final_hash: {result.expected_final_hash}")
    print(f"actual_final_hash: {result.actual_final_hash}")
    if args.output_trace_json:
        print(f"trace: {args.output_trace_json}")
    if result.error is not None:
        print(f"error: {result.error}")
    if result.final_text:
        print(f"final_text: {result.final_text}")
    if not result.passed:
        raise SystemExit(1)


def _run_task_batch(args: argparse.Namespace) -> None:
    tasks_dir = Path(args.tasks_dir)
    if args.task_ids:
        task_paths = [tasks_dir / f"{task_id}.json" for task_id in args.task_ids]
    else:
        task_paths = sorted(tasks_dir.glob("*.json"))
    if args.limit is not None:
        task_paths = task_paths[: args.limit]

    jsonl_handle = None
    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")

    results: list[dict[str, object]] = []
    passed = 0
    try:
        for task_path in task_paths:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_id = str(task.get("id", task_path.stem))
            try:
                result = run_task_with_openai_compatible(
                    task,
                    model=args.model,
                    endpoint=args.endpoint,
                    api_key_env=args.api_key_env,
                    retrieval_config=args.retrieval_config,
                    max_tool_steps=args.max_tool_steps,
                    timeout_seconds=args.timeout_seconds,
                )
                result_data = dataclasses.asdict(result)
                result_passed = result.passed
                stopped_reason = result.stopped_reason
                action_count = len(result.actions)
            except Exception as exc:  # noqa: BLE001 - one bad task should be reported.
                result_passed = False
                stopped_reason = "endpoint_error"
                action_count = 0
                result_data = {
                    "task_id": task_id,
                    "passed": False,
                    "actions": [],
                    "final_text": "",
                    "evaluation": None,
                    "stopped_reason": stopped_reason,
                    "expected_final_hash": None,
                    "actual_final_hash": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": {},
                }

            if result_passed:
                passed += 1
            results.append(result_data)
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                jsonl_handle.flush()
            print(
                f"task: {task_id} passed: {str(result_passed).lower()} "
                f"actions: {action_count} stopped_reason: {stopped_reason}",
                flush=True,
            )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    total = len(results)
    failed = total - passed
    pass_rate = passed / total if total else 0.0
    print(f"tasks: {total}")
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print(f"pass_rate: {pass_rate:.4f}")
    print(f"model: {args.model}")
    print(f"endpoint: {args.endpoint}")
    print(f"retrieval_config: {args.retrieval_config}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
