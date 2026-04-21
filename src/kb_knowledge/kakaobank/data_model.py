"""Transactional state models for the KakaoBank benchmark domain.

The shape intentionally mirrors the original tau3/tau2 ``banking_knowledge``
domain: a mutable transactional DB made of named tables, where each table has a
``data`` mapping and human-readable ``notes``. Retrieval documents stay outside
this DB and are handled by the fact/document export.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Iterator

from pydantic import BaseModel, ConfigDict, Field

ACTION_VERIFIER_SCHEMA_PATH = Path(
    "data/kakaobank_knowledge/v0/schema/action_verifier_state.json"
)
KAKAO_BANK_DB_SCHEMA_VERSION = "kakaobank_transactional_db.v0"

TAU3_BANKING_KNOWLEDGE_CROSSCHECK: tuple[dict[str, str], ...] = (
    {
        "topic": "database_table_shape",
        "source": "/Users/jd/Documents/workspace/tau3-bench/src/tau2/domains/banking_knowledge/data_model.py:80",
        "finding": "DatabaseTable stores mutable records under data and keeps table notes separately.",
        "decision": "KakaoBank DatabaseTable preserves the same data/notes shape.",
    },
    {
        "topic": "transactional_db_hash",
        "source": "/Users/jd/Documents/workspace/tau3-bench/src/tau2/environment/db.py:28",
        "finding": "Domain DB state is scored by hashing the Pydantic model dump.",
        "decision": "KakaoBankDB exposes get_hash() over a sorted JSON model dump.",
    },
    {
        "topic": "knowledge_domain_tables",
        "source": "/Users/jd/Documents/workspace/tau3-bench/src/tau2/domains/banking_knowledge/data_model.py:87",
        "finding": "banking_knowledge uses one TransactionalDB with named mutable tables.",
        "decision": "KakaoBankDB uses one transactional model with product-neutral named tables.",
    },
    {
        "topic": "in_memory_crud",
        "source": "/Users/jd/Documents/workspace/tau3-bench/src/tau2/domains/banking_knowledge/db_query.py:276",
        "finding": "banking_knowledge queries in-memory tables with operator suffix constraints.",
        "decision": "KakaoBank db_query keeps the same constraint vocabulary.",
    },
    {
        "topic": "schema_no_extra",
        "source": "/Users/jd/Documents/workspace/tau3-bench/src/tau2/utils/pydantic_utils.py:11",
        "finding": "tau3 DB models forbid extra fields through BaseModelNoExtra.",
        "decision": "KakaoBankDB and DatabaseTable reject unknown top-level fields.",
    },
)

TABLE_NAMES: tuple[str, ...] = (
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
    "transfer_references",
    "transactions",
    "disputes",
)


class BaseModelNoExtra(BaseModel):
    """Pydantic model base that rejects unknown fields like tau3's DB base."""

    model_config = ConfigDict(extra="forbid")


class DatabaseTable(BaseModelNoExtra):
    """A transactional DB table with records and optional notes.

    This cross-checks with
    ``tau2/domains/banking_knowledge/data_model.py:80`` in the source tau3 repo.
    """

    data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""


class KakaoBankDB(BaseModelNoExtra):
    """Mutable transactional DB for KakaoBank synthetic tasks.

    The table names are derived from
    ``data/kakaobank_knowledge/v0/schema/action_verifier_state.json`` and are deliberately
    separate from the retrieval KB documents generated from the fact DB.
    """

    schema_version: str = KAKAO_BANK_DB_SCHEMA_VERSION
    customers: DatabaseTable = Field(default_factory=DatabaseTable)
    businesses: DatabaseTable = Field(default_factory=DatabaseTable)
    consents: DatabaseTable = Field(default_factory=DatabaseTable)
    accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    deposit_contracts: DatabaseTable = Field(default_factory=DatabaseTable)
    savings_boxes: DatabaseTable = Field(default_factory=DatabaseTable)
    auto_transfer_rules: DatabaseTable = Field(default_factory=DatabaseTable)
    group_memberships: DatabaseTable = Field(default_factory=DatabaseTable)
    pockets: DatabaseTable = Field(default_factory=DatabaseTable)
    child_relationships: DatabaseTable = Field(default_factory=DatabaseTable)
    cards: DatabaseTable = Field(default_factory=DatabaseTable)
    card_orders: DatabaseTable = Field(default_factory=DatabaseTable)
    prepaid_wallets: DatabaseTable = Field(default_factory=DatabaseTable)
    loans: DatabaseTable = Field(default_factory=DatabaseTable)
    loan_applications: DatabaseTable = Field(default_factory=DatabaseTable)
    refinance_requests: DatabaseTable = Field(default_factory=DatabaseTable)
    required_documents: DatabaseTable = Field(default_factory=DatabaseTable)
    mortgage_collateral: DatabaseTable = Field(default_factory=DatabaseTable)
    lease_contracts: DatabaseTable = Field(default_factory=DatabaseTable)
    vehicle_purchase_cases: DatabaseTable = Field(default_factory=DatabaseTable)
    comparison_sessions: DatabaseTable = Field(default_factory=DatabaseTable)
    remittance_profiles: DatabaseTable = Field(default_factory=DatabaseTable)
    remittance_cases: DatabaseTable = Field(default_factory=DatabaseTable)
    service_enrollments: DatabaseTable = Field(default_factory=DatabaseTable)
    transfer_references: DatabaseTable = Field(default_factory=DatabaseTable)
    transactions: DatabaseTable = Field(default_factory=DatabaseTable)
    disputes: DatabaseTable = Field(default_factory=DatabaseTable)

    table_names: ClassVar[tuple[str, ...]] = TABLE_NAMES

    @classmethod
    def empty_with_schema_notes(
        cls,
        schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
    ) -> "KakaoBankDB":
        """Create an empty DB and populate table notes from the schema file."""

        db = cls()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for table in schema["state_tables"]:
            table_name = table["name"]
            if table_name not in cls.table_names:
                continue
            getattr(db, table_name).notes = _format_table_notes(table)
        return db

    @classmethod
    def load(cls, path: str | Path) -> "KakaoBankDB":
        """Load a KakaoBankDB from a JSON file."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def dump(self, path: str | Path, *, exclude_defaults: bool = False) -> None:
        """Write a KakaoBankDB to a JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                self.model_dump(exclude_defaults=exclude_defaults),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def iter_tables(self) -> Iterator[tuple[str, DatabaseTable]]:
        """Yield table names and table objects in deterministic schema order."""

        for table_name in self.table_names:
            yield table_name, getattr(self, table_name)

    def get_table(self, table_name: str) -> DatabaseTable | None:
        """Return a table by name, or None when the table is not defined."""

        if table_name not in self.table_names:
            return None
        return getattr(self, table_name)

    def get_hash(self) -> str:
        """Return a stable hash for tau3-style DB state comparison."""

        payload = self.model_dump()
        hash_string = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(hash_string.encode("utf-8")).hexdigest()

    def get_statistics(self) -> dict[str, int]:
        """Return record counts for every transactional table."""

        return {
            f"num_{table_name}": len(table.data)
            for table_name, table in self.iter_tables()
        }

    def validate_against_action_schema(
        self,
        schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
    ) -> list[str]:
        """Check that this DB's tables match the action/verifier schema."""

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_tables = {table["name"] for table in schema["state_tables"]}
        model_tables = set(self.table_names)
        errors: list[str] = []

        missing = sorted(schema_tables - model_tables)
        extra = sorted(model_tables - schema_tables)
        if missing:
            errors.append(f"Missing DB tables from schema: {missing}")
        if extra:
            errors.append(f"DB tables not declared in schema: {extra}")

        for table_name, table in self.iter_tables():
            if not isinstance(table.data, dict):
                errors.append(f"{table_name}.data must be a dict")
            for record_id, record in table.data.items():
                if not isinstance(record_id, str):
                    errors.append(f"{table_name} record id must be str: {record_id!r}")
                if not isinstance(record, dict):
                    errors.append(f"{table_name}.{record_id} record must be a dict")

        return errors


def load_action_verifier_schema(
    schema_path: Path = ACTION_VERIFIER_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load the action/verifier schema that defines the DB table set."""

    return json.loads(schema_path.read_text(encoding="utf-8"))


def _format_table_notes(table_schema: dict[str, Any]) -> str:
    fields = ", ".join(table_schema.get("required_fields", []))
    verifier_use = "; ".join(table_schema.get("verifier_use", []))
    return (
        f"purpose: {table_schema.get('purpose', '')}\n"
        f"key: {table_schema.get('key', '')}\n"
        f"required_fields: {fields}\n"
        f"verifier_use: {verifier_use}"
    )
