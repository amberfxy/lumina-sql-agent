"""FastMCP server exposing LuminaSQL as typed MCP tools.

Usage:
    python -m mcp_server.server                                  # stdio (Claude Desktop, Claude Code, IDEs)
    python -m mcp_server.server --transport http --port 8765     # streamable HTTP at /mcp
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, TokenVerifier
from pydantic import Field, StringConstraints

from config import DatabaseBackend, get_settings
from mcp_server.tools import AskResult, LuminaMCPService, QueryResult, SchemaInfo, ServiceUnavailable, ValidationResult
from src.admission import AdmissionController, AdmissionRejected
from src.context import configure_logging, new_request_id, reset_request_id, set_request_id

logger = logging.getLogger(__name__)

T = TypeVar("T")

SQL = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=10000),
    Field(description="A single read-only PostgreSQL SELECT (or PartiQL SELECT for the dynamodb backend)."),
]
Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    Field(description="A natural-language question about the data."),
]
Backend = Annotated[DatabaseBackend, Field(description="Database to target.")]

_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}

INSTRUCTIONS = """LuminaSQL answers questions about a relational database through read-only queries.
Typical flow: call get_schema once, then either ask_database with the user's question, or write SQL
yourself and call validate_sql followed by execute_readonly_query. Every query passes the same
AST validator, READ ONLY transaction, and least-privilege database role as the LuminaSQL REST API;
data-modifying statements are always rejected. A failure's `policy` says whether retrying can help:
`llm_correctable` (fix the query and retry), `transient` (retry as-is later), `terminal` (do not retry)."""


class ApiKeyVerifier(TokenVerifier):
    """Bearer-token auth for HTTP transports, using the same API_KEYS as the REST API."""

    def __init__(self, keys: set[str]) -> None:
        super().__init__()
        self._keys = keys

    async def verify_token(self, token: str) -> AccessToken | None:
        if not any(hmac.compare_digest(token, key) for key in self._keys):
            return None
        return AccessToken(token=token, client_id="key:" + token[-6:], scopes=[])


def create_server(service: LuminaMCPService | None = None, *, auth: TokenVerifier | None = None) -> FastMCP:
    service = service or LuminaMCPService()
    settings = service.settings
    # Same bound as the REST API: each query holds a worker thread and a DB connection.
    admission = AdmissionController(
        settings.max_concurrent_requests, settings.max_queued_requests, settings.queue_timeout_seconds
    )
    mcp = FastMCP(
        name="LuminaSQL",
        instructions=INSTRUCTIONS,
        auth=auth,
        # Unexpected exceptions reach the client as a generic error; ToolError messages are kept.
        mask_error_details=True,
    )

    async def call(fn: Callable[..., T], *args: Any, admit: bool = False) -> T:
        token = set_request_id(new_request_id())
        try:
            if admit:
                try:
                    await admission.acquire()
                except AdmissionRejected as exc:
                    raise ToolError(f"Server busy ({exc.reason}); retry later.") from exc
            try:
                # Blocking DB/LLM work runs in a thread; to_thread carries the request ID along.
                return await asyncio.to_thread(fn, *args)
            except ServiceUnavailable as exc:
                raise ToolError(str(exc)) from exc
            finally:
                if admit:
                    admission.release()
        finally:
            reset_request_id(token)

    @mcp.tool(annotations={**_READ_ONLY, "idempotentHint": True})
    async def get_schema(backend: Backend = DatabaseBackend.POSTGRES) -> SchemaInfo:
        """List the tables and columns you may query, with types, keys, foreign keys, and table descriptions.

        Tables outside the server's allowlist and restricted columns are omitted. Call this before
        writing SQL yourself.
        """
        return await call(service.get_schema, backend)

    @mcp.tool(annotations={**_READ_ONLY, "idempotentHint": True})
    async def validate_sql(sql: SQL, backend: Backend = DatabaseBackend.POSTGRES) -> ValidationResult:
        """Check a query against LuminaSQL's safety and schema rules without executing it.

        Returns valid=false with a reason and failure category for anything other than a single
        read-only SELECT over known tables (writes, DDL, multiple statements, system catalogs,
        side-effecting functions, restricted columns, unknown tables, or syntax errors).
        """
        return await call(service.validate_sql, sql, backend)

    @mcp.tool(annotations=_READ_ONLY)
    async def execute_readonly_query(sql: SQL, backend: Backend = DatabaseBackend.POSTGRES) -> QueryResult:
        """Validate and execute one read-only query, returning columns and rows.

        Unsafe or invalid queries are rejected before reaching the database. Results are capped at the
        server's row limit (truncated=true when more rows exist) and bounded by a statement timeout.
        """
        return await call(service.execute_readonly_query, sql, backend, admit=True)

    @mcp.tool(annotations=_READ_ONLY)
    async def ask_database(question: Question, backend: Backend = DatabaseBackend.POSTGRES) -> AskResult:
        """Answer a natural-language question end to end: generate SQL, validate, execute, and
        self-correct on fixable errors (syntax, unknown table/column, type errors).

        Returns the generated and final SQL, each failed attempt with its category, and the result rows.
        Requests to modify data are refused.
        """
        return await call(service.ask_database, question, backend, admit=True)

    return mcp


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LuminaSQL MCP server")
    parser.add_argument("--transport", choices=["stdio", "http", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP transports only")
    parser.add_argument("--port", type=int, default=8765, help="HTTP transports only")
    args = parser.parse_args(argv)

    settings = get_settings()
    # Logs go to stderr; stdout carries the stdio protocol stream.
    configure_logging(settings.log_level, settings.log_format)
    service = LuminaMCPService(settings)
    auth = None
    transport_kwargs: dict[str, Any] = {}
    if args.transport != "stdio":
        transport_kwargs = {"host": args.host, "port": args.port}
        if settings.api_key_set:
            auth = ApiKeyVerifier(settings.api_key_set)
        elif not _is_loopback(args.host):
            raise SystemExit("Refusing to serve MCP over HTTP on a non-loopback address without API_KEYS.")
    try:
        create_server(service, auth=auth).run(transport=args.transport, show_banner=False, **transport_kwargs)
    finally:
        service.close()


if __name__ == "__main__":
    main()
