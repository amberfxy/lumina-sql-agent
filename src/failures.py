"""Failure taxonomy and retry policy.

Every failure in the agent loop is classified into one category, and each
category maps to exactly one policy:

- LLM_CORRECTABLE: the query was wrong in a way the model can fix from the error
  message (syntax, unknown column, type mismatch). The error goes back to the LLM.
- TRANSIENT: the infrastructure hiccuped (connection drop, pool exhaustion,
  throttling). The same query is retried with backoff; the LLM is not involved.
- TERMINAL: retrying cannot help or must not happen (unsafe query, permission
  denied, timeout, provider outage). The run stops immediately.

Unsafe queries are deliberately terminal: feeding a safety rejection back to the
model would turn it into an optimizer against the guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError


class FailureCategory(StrEnum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    UNKNOWN_TABLE = "UNKNOWN_TABLE"
    UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
    SEMANTIC_ERROR = "SEMANTIC_ERROR"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    UNSAFE_QUERY = "UNSAFE_QUERY"
    UNANSWERABLE = "UNANSWERABLE"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    MODEL_ERROR = "MODEL_ERROR"
    SCHEMA_UNAVAILABLE = "SCHEMA_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class RetryPolicy(StrEnum):
    LLM_CORRECTABLE = "llm_correctable"
    TRANSIENT = "transient"
    TERMINAL = "terminal"


class FailureSource(StrEnum):
    VALIDATOR = "validator"
    DATABASE = "database"
    LLM = "llm"
    AGENT = "agent"


_LLM_CORRECTABLE = {
    FailureCategory.SYNTAX_ERROR,
    FailureCategory.UNKNOWN_TABLE,
    FailureCategory.UNKNOWN_COLUMN,
    FailureCategory.SEMANTIC_ERROR,
    FailureCategory.MALFORMED_OUTPUT,
}


def policy_for(category: FailureCategory, source: FailureSource) -> RetryPolicy:
    if category in _LLM_CORRECTABLE:
        return RetryPolicy.LLM_CORRECTABLE
    # The LLM SDKs already retry 429/5xx/timeouts with backoff; once they give up, so do we.
    if source == FailureSource.DATABASE and category in (FailureCategory.CONNECTION_ERROR, FailureCategory.RATE_LIMIT):
        return RetryPolicy.TRANSIENT
    return RetryPolicy.TERMINAL


@dataclass(frozen=True)
class Failure:
    category: FailureCategory
    source: FailureSource
    message: str

    @property
    def policy(self) -> RetryPolicy:
        return policy_for(self.category, self.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "source": self.source.value,
            "policy": self.policy.value,
            "message": self.message,
        }


# PostgreSQL SQLSTATE codes: https://www.postgresql.org/docs/current/errcodes-appendix.html
_PG_EXACT = {
    "42601": FailureCategory.SYNTAX_ERROR,
    "42P01": FailureCategory.UNKNOWN_TABLE,
    "42703": FailureCategory.UNKNOWN_COLUMN,
    "42501": FailureCategory.PERMISSION_ERROR,
    "25006": FailureCategory.UNSAFE_QUERY,  # read_only_sql_transaction
    "57014": FailureCategory.TIMEOUT,  # query_canceled (statement_timeout)
    "40001": FailureCategory.CONNECTION_ERROR,  # serialization_failure: retry as-is
    "40P01": FailureCategory.CONNECTION_ERROR,  # deadlock_detected: retry as-is
}
_PG_CLASS = {
    "08": FailureCategory.CONNECTION_ERROR,  # connection exception
    "53": FailureCategory.CONNECTION_ERROR,  # insufficient resources (too many connections)
    "57": FailureCategory.CONNECTION_ERROR,  # operator intervention (admin shutdown, crash recovery)
    "42": FailureCategory.SEMANTIC_ERROR,  # remaining syntax/access rule violations (types, grouping, ambiguity)
    "22": FailureCategory.SEMANTIC_ERROR,  # data exception (division by zero, bad cast)
    "0A": FailureCategory.SEMANTIC_ERROR,  # feature not supported
    "54": FailureCategory.SEMANTIC_ERROR,  # program limit exceeded (query too complex)
}


def classify_postgres_error(exc: SQLAlchemyError) -> FailureCategory:
    if isinstance(exc, PoolTimeoutError):
        return FailureCategory.CONNECTION_ERROR
    pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
    if pgcode:
        if pgcode in _PG_EXACT:
            return _PG_EXACT[pgcode]
        if pgcode[:2] in _PG_CLASS:
            return _PG_CLASS[pgcode[:2]]
        return FailureCategory.INTERNAL_ERROR
    # No SQLSTATE: the server never answered (refused, DNS, dropped connection).
    if isinstance(exc, OperationalError) or (isinstance(exc, DBAPIError) and exc.connection_invalidated):
        return FailureCategory.CONNECTION_ERROR
    return FailureCategory.INTERNAL_ERROR


_DYNAMODB_CODES = {
    "ValidationException": FailureCategory.SYNTAX_ERROR,
    "ResourceNotFoundException": FailureCategory.UNKNOWN_TABLE,
    "AccessDeniedException": FailureCategory.PERMISSION_ERROR,
    "UnrecognizedClientException": FailureCategory.PERMISSION_ERROR,
    "ProvisionedThroughputExceededException": FailureCategory.RATE_LIMIT,
    "ThrottlingException": FailureCategory.RATE_LIMIT,
    "RequestLimitExceeded": FailureCategory.RATE_LIMIT,
    "InternalServerError": FailureCategory.CONNECTION_ERROR,
    "ServiceUnavailable": FailureCategory.CONNECTION_ERROR,
}


def classify_dynamodb_error(exc: BotoCoreError | ClientError) -> FailureCategory:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return _DYNAMODB_CODES.get(code, FailureCategory.INTERNAL_ERROR)
    if isinstance(exc, ReadTimeoutError):
        return FailureCategory.TIMEOUT
    if isinstance(exc, (EndpointConnectionError, ConnectTimeoutError)):
        return FailureCategory.CONNECTION_ERROR
    return FailureCategory.INTERNAL_ERROR


def classify_llm_error(exc: Exception) -> FailureCategory:
    """Map OpenAI/Anthropic SDK exceptions by name so both SDKs share one code path."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "RateLimitError" in names:
        return FailureCategory.RATE_LIMIT
    if "APITimeoutError" in names or isinstance(exc, TimeoutError):
        return FailureCategory.TIMEOUT
    return FailureCategory.MODEL_ERROR
