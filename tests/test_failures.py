from __future__ import annotations

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from eval.scripted_llm import raise_fault
from src.failures import (
    FailureCategory,
    FailureSource,
    RetryPolicy,
    classify_dynamodb_error,
    classify_llm_error,
    classify_postgres_error,
    policy_for,
)


class _PgError(Exception):
    def __init__(self, pgcode: str | None) -> None:
        super().__init__("pg error")
        self.pgcode = pgcode


@pytest.mark.parametrize(
    ("pgcode", "category"),
    [
        ("42601", FailureCategory.SYNTAX_ERROR),
        ("42P01", FailureCategory.UNKNOWN_TABLE),
        ("42703", FailureCategory.UNKNOWN_COLUMN),
        ("42883", FailureCategory.SEMANTIC_ERROR),  # undefined function / operator
        ("42702", FailureCategory.SEMANTIC_ERROR),  # ambiguous column
        ("42803", FailureCategory.SEMANTIC_ERROR),  # grouping error
        ("22012", FailureCategory.SEMANTIC_ERROR),  # division by zero
        ("42501", FailureCategory.PERMISSION_ERROR),
        ("25006", FailureCategory.UNSAFE_QUERY),
        ("57014", FailureCategory.TIMEOUT),
        ("08006", FailureCategory.CONNECTION_ERROR),
        ("53300", FailureCategory.CONNECTION_ERROR),
        ("57P01", FailureCategory.CONNECTION_ERROR),
        ("XX000", FailureCategory.INTERNAL_ERROR),
    ],
)
def test_postgres_sqlstate_classification(pgcode, category):
    assert classify_postgres_error(ProgrammingError("q", {}, _PgError(pgcode))) == category


def test_postgres_errors_without_sqlstate():
    assert classify_postgres_error(OperationalError("q", {}, _PgError(None))) == FailureCategory.CONNECTION_ERROR
    assert classify_postgres_error(PoolTimeoutError("pool exhausted")) == FailureCategory.CONNECTION_ERROR


def test_dynamodb_classification():
    def client_error(code: str) -> ClientError:
        return ClientError({"Error": {"Code": code, "Message": "m"}}, "ExecuteStatement")

    assert classify_dynamodb_error(client_error("ValidationException")) == FailureCategory.SYNTAX_ERROR
    assert classify_dynamodb_error(client_error("ResourceNotFoundException")) == FailureCategory.UNKNOWN_TABLE
    assert classify_dynamodb_error(client_error("ThrottlingException")) == FailureCategory.RATE_LIMIT
    assert classify_dynamodb_error(client_error("AccessDeniedException")) == FailureCategory.PERMISSION_ERROR
    assert classify_dynamodb_error(EndpointConnectionError(endpoint_url="x")) == FailureCategory.CONNECTION_ERROR
    assert classify_dynamodb_error(ReadTimeoutError(endpoint_url="x")) == FailureCategory.TIMEOUT


@pytest.mark.parametrize(
    ("marker", "category"),
    [
        ("!RATE_LIMIT", FailureCategory.RATE_LIMIT),
        ("!TIMEOUT", FailureCategory.TIMEOUT),
        ("!SERVER_ERROR", FailureCategory.MODEL_ERROR),
    ],
)
def test_llm_sdk_exception_classification(marker, category):
    with pytest.raises(Exception) as excinfo:
        raise_fault(marker)
    assert classify_llm_error(excinfo.value) == category


def test_policies():
    db, llm, validator = FailureSource.DATABASE, FailureSource.LLM, FailureSource.VALIDATOR
    assert policy_for(FailureCategory.UNKNOWN_COLUMN, db) == RetryPolicy.LLM_CORRECTABLE
    assert policy_for(FailureCategory.MALFORMED_OUTPUT, validator) == RetryPolicy.LLM_CORRECTABLE
    assert policy_for(FailureCategory.CONNECTION_ERROR, db) == RetryPolicy.TRANSIENT
    assert policy_for(FailureCategory.RATE_LIMIT, db) == RetryPolicy.TRANSIENT
    assert policy_for(FailureCategory.RATE_LIMIT, llm) == RetryPolicy.TERMINAL
    assert policy_for(FailureCategory.UNSAFE_QUERY, validator) == RetryPolicy.TERMINAL
    assert policy_for(FailureCategory.TIMEOUT, db) == RetryPolicy.TERMINAL
    assert policy_for(FailureCategory.PERMISSION_ERROR, db) == RetryPolicy.TERMINAL
