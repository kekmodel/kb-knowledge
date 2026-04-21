# kb-knowledge

KakaoBank knowledge benchmark v0 evaluator and OpenAI-compatible assistant
runner.

This repository contains only the assets needed to replay and evaluate the
current KakaoBank DB-delta task set. It is not affiliated with KakaoBank.

## Contents

```text
data/
  kakaobank_knowledge/
    v0/
      schema/
        action_verifier_state.json

      tasks/
        db.json
        tasks.json
        tasks.summary.md
        cases/
          *.json

      knowledge_base/
        INDEX.md
        documents/
          *.json

src/
  kb_knowledge/
    cli.py
    kakaobank/
      data_model.py
      db_query.py
      replay.py
      runner.py
      tools.py
```

## What Each Data Folder Means

`schema/` is the evaluator contract. It defines runtime DB tables, action
families, tool requestors, mutation flags, and verifier-related metadata. Both
DB replay and model evaluation need it.

`tasks/` is the benchmark task set. `tasks/cases/` contains the 123 per-task JSON
files used by the CLI. `tasks.json` is the aggregate export, and `db.json` is the
empty tau-style domain DB fixture.

`knowledge_base/` is the assistant search corpus. `KB_search` and `grep` load
the 207 JSON documents from `knowledge_base/documents/`. DB-only replay does not
need these documents, but actual model evaluation does.

## Setup

```bash
uv sync
```

## DB Replay

Replay all 123 exported DB-delta tasks deterministically:

```bash
uv run kb-knowledge replay-kakaobank-tasks
```

Expected result:

```text
tasks: 123
passed: 123
failed: 0
```

Evaluate captured assistant tool calls for one task:

```bash
uv run kb-knowledge evaluate-kakaobank-actions \
  --task-id kb_manual_demand_deposit_clean_close_success \
  --actions-json candidate_actions.json
```

## Model Evaluation

Run one task through OpenAI API, vLLM, SGLang, or another OpenAI-compatible
chat-completions endpoint:

```bash
uv run kb-knowledge run-kakaobank-task \
  --task-id kb_manual_demand_deposit_clean_close_success \
  --model served-model \
  --endpoint http://localhost:8000/v1 \
  --retrieval-config bm25_grep \
  --output-trace-json logs/trace.json
```

Run a batch:

```bash
uv run kb-knowledge run-kakaobank-tasks \
  --model served-model \
  --endpoint http://localhost:8000/v1 \
  --retrieval-config bm25_grep \
  --output-json logs/results.json \
  --output-jsonl logs/episodes.jsonl
```

`--endpoint` may be a full `/chat/completions` URL or an OpenAI-style `/v1` base
URL. `OPENAI_API_KEY` is used when present; local compatible servers that do not
require authentication work without it.

`--retrieval-config` defaults to `bm25_grep` and also supports `bm25` and
`grep`. Text embedding and terminal-use variants are intentionally excluded from
this v0 runner.

## Evaluation Semantics

The v0 runner receives an initial `user_prompt`, lets the assistant chain tools,
and grades the final DB state. A run passes only when:

1. replaying the assistant write actions produces the expected final DB hash
2. the assistant terminates with the `done` tool

Plain assistant text is not a completion signal in this runner.
