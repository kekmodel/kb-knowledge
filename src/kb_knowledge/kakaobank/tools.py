"""Minimal KakaoBank v0 tool implementations.

These are standalone deterministic helpers for the local v0 replay/runner work.
They do not yet implement tau2 ``ToolKitBase`` integration.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from rank_bm25 import BM25Okapi

from kb_knowledge.kakaobank.data_model import KakaoBankDB
from kb_knowledge.kakaobank.db_query import query_db

KAKAOBANK_DOCUMENTS_DIR = Path(
    "data/kakaobank_knowledge/v0/knowledge_base/documents"
)
SUPPORTED_RETRIEVAL_CONFIGS = ("bm25", "grep", "bm25_grep")
DEFAULT_RETRIEVAL_CONFIG = "bm25_grep"
BM25_RETRIEVAL_CONFIGS = {"bm25", "bm25_grep"}
GREP_RETRIEVAL_CONFIGS = {"grep", "bm25_grep"}


@dataclass(frozen=True)
class KakaoBankToolDefinition:
    """Small v0 tool metadata record until the tau2 ToolKitBase layer exists."""

    name: str
    requestor: Literal["assistant", "user"]
    tool_type: Literal["read", "write"]
    mutates_state: bool
    parameters: tuple[str, ...]
    description: str


ASSISTANT_READ_TOOL_DEFINITIONS: tuple[KakaoBankToolDefinition, ...] = (
    KakaoBankToolDefinition(
        name="KB_search",
        requestor="assistant",
        tool_type="read",
        mutates_state=False,
        parameters=("query",),
        description="Search KakaoBank product and policy knowledge documents with BM25.",
    ),
    KakaoBankToolDefinition(
        name="grep",
        requestor="assistant",
        tool_type="read",
        mutates_state=False,
        parameters=("pattern",),
        description="Search KakaoBank knowledge documents with a regex or literal pattern.",
    ),
    KakaoBankToolDefinition(
        name="get_customer_profile",
        requestor="assistant",
        tool_type="read",
        mutates_state=False,
        parameters=("customer_id",),
        description="Read customer, business, and consent records for one customer.",
    ),
    KakaoBankToolDefinition(
        name="get_account_or_contract",
        requestor="assistant",
        tool_type="read",
        mutates_state=False,
        parameters=("record_id", "table"),
        description="Read one runtime DB record by table and record_id.",
    ),
)


class KakaoBankReadTools:
    """Read-only v0 tools backed by KakaoBankDB and exported KB documents."""

    def __init__(
        self,
        db: KakaoBankDB,
        *,
        documents_dir: Path = KAKAOBANK_DOCUMENTS_DIR,
        retrieval_config: str = DEFAULT_RETRIEVAL_CONFIG,
    ):
        if retrieval_config not in SUPPORTED_RETRIEVAL_CONFIGS:
            raise ValueError(
                f"unknown retrieval_config: {retrieval_config!r}; "
                f"supported: {', '.join(SUPPORTED_RETRIEVAL_CONFIGS)}"
            )
        self.db = db
        self.retrieval_config = retrieval_config
        self.documents = load_knowledge_documents(documents_dir)
        self.search_index = BM25DocumentIndex(self.documents)
        self.grep_index = GrepDocumentIndex(self.documents)

    def KB_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return deterministic assistant-facing KB search results."""

        documents = self.search_index.search(query, top_k=top_k)
        return {
            "query": query,
            "documents": documents,
            "missing_document_ids": [],
        }

    def grep(
        self,
        pattern: str,
        top_k: int = 10,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Return deterministic regex/literal search results over KB documents."""

        matches = self.grep_index.search(
            pattern,
            top_k=top_k,
            case_sensitive=case_sensitive,
        )
        return {
            "pattern": pattern,
            "documents": matches,
        }

    def get_customer_profile(self, customer_id: str) -> dict[str, Any]:
        """Return customer, business, and consent records for one customer."""

        customer = _get_record_by_id_or_field(
            self.db,
            table_name="customers",
            record_id=customer_id,
            field_name="customer_id",
        )
        if customer is None:
            raise ValueError(f"customer not found: {customer_id}")
        businesses = query_db("businesses", self.db, customer_id=customer_id)
        consents = query_db("consents", self.db, customer_id=customer_id)
        return {
            "customer_id": customer_id,
            "customer": deepcopy(customer),
            "businesses": deepcopy(businesses),
            "consents": deepcopy(consents),
        }

    def get_account_or_contract(
        self,
        record_id: str,
        table: str,
    ) -> dict[str, Any]:
        """Return one runtime DB record by table and record_id."""

        record = _get_record_by_id_or_field(
            self.db,
            table_name=table,
            record_id=record_id,
        )
        if record is None:
            raise ValueError(f"record not found in {table}: {record_id}")
        return {
            "table": table,
            "record_id": record_id,
            "record": deepcopy(record),
        }


def load_knowledge_documents(
    documents_dir: Path = KAKAOBANK_DOCUMENTS_DIR,
) -> dict[str, dict[str, str]]:
    """Load exported ``id/title/content`` KB documents by ID."""

    documents: dict[str, dict[str, str]] = {}
    for path in sorted(documents_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        document = {
            "id": str(data["id"]),
            "title": str(data["title"]),
            "content": str(data["content"]),
        }
        documents[document["id"]] = document
    return documents


def get_assistant_read_tool_definitions() -> tuple[KakaoBankToolDefinition, ...]:
    """Return assistant-facing read tool metadata for the v0 runner."""

    return ASSISTANT_READ_TOOL_DEFINITIONS


class BM25DocumentIndex:
    """In-memory BM25 index over the fair title+content KB search surface."""

    def __init__(self, documents: dict[str, dict[str, str]]) -> None:
        self.documents = documents
        self._indexed_documents = list(documents.items())
        tokenized_corpus = [
            _document_search_text(document).lower().split()
            for _, document in self._indexed_documents
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        """Return top documents by BM25 scoring over title+content."""

        if not query or not query.strip():
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_k = min(top_k, len(self._indexed_documents))
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for index in sorted_indices:
            _, document = self._indexed_documents[index]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "content": document["content"],
                    "score": float(scores[index]),
                }
            )
        return results


KnowledgeDocumentIndex = BM25DocumentIndex


class GrepDocumentIndex:
    """Regex/literal search index over the fair title+content KB surface."""

    def __init__(self, documents: dict[str, dict[str, str]]) -> None:
        self.documents = documents
        self._indexed_documents = [
            (doc_id, document, _document_search_text(document))
            for doc_id, document in documents.items()
        ]

    def search(
        self,
        pattern: str,
        *,
        top_k: int = 10,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_pattern = pattern.strip()
        if not normalized_pattern:
            return []

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(normalized_pattern, flags)
        except re.error:
            compiled = re.compile(re.escape(normalized_pattern), flags)

        scored: list[tuple[int, str, dict[str, Any]]] = []
        for doc_id, document, haystack in self._indexed_documents:
            match_count = len(compiled.findall(haystack))
            if match_count == 0:
                continue
            scored.append(
                (
                    match_count,
                    doc_id,
                    {
                        "id": document["id"],
                        "title": document["title"],
                        "content": document["content"],
                        "score": float(match_count),
                        "match_count": match_count,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, _, document in scored[:top_k]]


def lookup_required_documents(
    documents: dict[str, dict[str, str]],
    *,
    query: str,
    required_document_ids: Sequence[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Return gold required documents without exposing oracle IDs as a tool arg."""

    matched_documents: list[dict[str, str]] = []
    missing_document_ids: list[str] = []

    for doc_id in required_document_ids:
        document = documents.get(doc_id)
        if document is None:
            missing_document_ids.append(doc_id)
            continue
        matched_documents.append(document)

    return {
        "query": query,
        "documents": matched_documents[:top_k],
        "missing_document_ids": missing_document_ids,
    }


def _get_record_by_id_or_field(
    db: KakaoBankDB,
    *,
    table_name: str,
    record_id: str,
    field_name: str | None = None,
) -> dict[str, Any] | None:
    table_data = db.get_table(table_name)
    if table_data is None:
        raise ValueError(f"unknown table: {table_name}")
    if record_id in table_data.data:
        return table_data.data[record_id]
    if field_name is not None:
        matches = query_db(table_name, db, **{field_name: record_id})
        if matches:
            return matches[0]
    return None


def _lexical_search(
    query: str,
    documents: dict[str, dict[str, str]],
    *,
    top_k: int,
) -> list[dict[str, str]]:
    return BM25DocumentIndex(documents).search(query, top_k=top_k)


def _document_search_text(document: dict[str, str]) -> str:
    """Return the searchable text surface, matching terminal-use markdown files."""

    return f"# {document['title']}\n\n{document['content']}"


def _tokenize(value: str) -> list[str]:
    return [
        token for token in re.split(r"""[\s`.,:;()\[\]{}<>"'/_-]+""", value) if token
    ]
