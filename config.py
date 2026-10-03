"""Centralized environment configuration for LuminaSQL-Agent."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class DatabaseBackend(StrEnum):
    POSTGRES = "postgres"
    DYNAMODB = "dynamodb"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "LuminaSQL-Agent"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_retry_iterations: int = Field(default=3, ge=1, le=10)
    max_result_rows: int = Field(default=500, ge=1, le=10000)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Must exceed the idle timeout of the load balancer/ingress in front of the API, otherwise
    # the server can close a connection the proxy is about to reuse (intermittent 502s).
    api_keepalive_timeout_seconds: int = Field(default=75, ge=1)
    # Created by the Kubernetes preStop hook; while it exists the API asks clients to close
    # keep-alive connections and reports not-ready, so traffic moves off before SIGTERM.
    drain_file: str = "/tmp/lumina-draining"
    api_base_url: str = "http://localhost:8000"
    cors_allow_origins: list[str] = ["*"]
    log_format: Literal["text", "json"] = "text"
    agent_deadline_seconds: float = Field(default=90.0, gt=0)

    # Authorization and safety
    api_keys: SecretStr | None = None  # comma-separated; unset disables API authentication
    mutations_enabled: bool = False  # server-side gate for the per-request allow_mutations flag
    allowed_tables: list[str] = []  # empty = every table in the configured schema
    denied_columns: list[str] = []  # "table.column" entries hidden from the model and rejected by the guard

    # Admission control and rate limiting
    max_concurrent_requests: int = Field(default=16, ge=1, le=1000)
    max_queued_requests: int = Field(default=32, ge=0, le=10000)
    queue_timeout_seconds: float = Field(default=10.0, gt=0)
    rate_limit_per_minute: int = Field(default=0, ge=0)  # per client; 0 disables

    # Transient database retries (same query, no LLM)
    db_transient_retries: int = Field(default=2, ge=0, le=5)
    db_retry_backoff_seconds: float = Field(default=0.2, ge=0)

    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    openai_base_url: str | None = None
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=16384)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    # USD per 1M tokens, used only for cost estimates (defaults: gpt-4o list price).
    llm_input_cost_per_1m_tokens: float = Field(default=2.50, ge=0)
    llm_output_cost_per_1m_tokens: float = Field(default=10.00, ge=0)

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("postgres")
    postgres_db: str = "lumina"
    postgres_schema: str = "public"
    postgres_sslmode: str = "prefer"
    postgres_pool_size: int = Field(default=10, ge=1, le=100)
    postgres_max_overflow: int = Field(default=10, ge=0, le=100)
    postgres_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    postgres_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    postgres_statement_timeout_ms: int = Field(default=15000, ge=100)

    # AWS / DynamoDB
    aws_region: str = "us-east-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    dynamodb_endpoint_url: str | None = None
    dynamodb_table_prefix: str = ""
    # Each PartiQL page reads up to 1 MB, so this bounds the read capacity a single query can consume.
    dynamodb_max_pages: int = Field(default=10, ge=1, le=100)

    # Redis cache (optional; caching is disabled when unset)
    redis_url: str | None = None
    schema_cache_ttl_seconds: int = Field(default=600, ge=1)
    query_cache_ttl_seconds: int = Field(default=3600, ge=1)
    redis_socket_timeout_seconds: float = Field(default=0.5, gt=0)
    redis_failure_cooldown_seconds: float = Field(default=30.0, ge=0)

    @property
    def postgres_url(self) -> str:
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )

    @property
    def api_key_set(self) -> set[str]:
        if not self.api_keys:
            return set()
        return {key.strip() for key in self.api_keys.get_secret_value().split(",") if key.strip()}

    @property
    def active_llm_model(self) -> str:
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model
        return self.openai_model


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
