from __future__ import annotations

from config import DatabaseBackend
from src.schema_manager import ColumnMetadata, SchemaCatalog, SchemaManager, TableMetadata


def table(name: str, *columns: str, description: str = "") -> TableMetadata:
    return TableMetadata(
        name=name,
        columns=tuple(ColumnMetadata(name=column, data_type="INTEGER") for column in columns),
        description=description,
    )


CATALOG = SchemaCatalog(
    backend=DatabaseBackend.POSTGRES,
    tables=[
        table("customers", "customer_id", "city", "segment"),
        table("orders", "order_id", "customer_id", "order_date", "status"),
        table("employees", "employee_id", "salary", "department"),
    ],
)


class CountingManager(SchemaManager):
    def __init__(self, settings, cache=None):
        super().__init__(settings=settings, db_executor=object(), cache=cache)
        self.extractions = 0

    def _extract_postgres_schema(self) -> SchemaCatalog:
        self.extractions += 1
        return CATALOG


def test_rank_tables_prefers_lexical_overlap(settings):
    manager = CountingManager(settings)
    ranked = manager._rank_tables("average salary by department", CATALOG.tables)
    assert ranked[0][0].name == "employees"


def test_pruned_schema_keeps_relevant_tables_only(settings):
    schema = CountingManager(settings).get_pruned_schema("orders status by customer")
    assert "`orders`" in schema
    assert "`customers`" in schema
    assert "`employees`" not in schema


def test_falls_back_to_top_tables_when_nothing_matches(settings):
    schema = CountingManager(settings).get_pruned_schema("zzz qqq")
    assert schema.count("### Table") == 3


def test_tokenizer_splits_identifiers_and_folds_plurals():
    tokens = SchemaManager._tokenize("customers customer_id categories")
    assert tokens.count("customer") == 2
    assert "category" in tokens


def test_catalog_json_round_trip():
    restored = SchemaCatalog.from_json(CATALOG.to_json())
    assert restored == CATALOG


def test_local_cache_avoids_repeated_extraction(settings):
    manager = CountingManager(settings)
    manager.get_pruned_schema("orders")
    manager.get_pruned_schema("customers")
    assert manager.extractions == 1


def test_shared_redis_catalog_is_reused_across_instances(settings, redis_cache):
    first = CountingManager(settings, cache=redis_cache)
    first.get_pruned_schema("orders")
    second = CountingManager(settings, cache=redis_cache)
    second.get_pruned_schema("orders")

    assert first.extractions == 1
    assert second.extractions == 0


def test_invalidate_cache_forces_reextraction(settings, redis_cache):
    manager = CountingManager(settings, cache=redis_cache)
    manager.get_pruned_schema("orders")
    manager.invalidate_cache()
    manager.get_pruned_schema("orders")
    assert manager.extractions == 2


def test_cosine_similarity_bounds():
    from collections import Counter

    assert SchemaManager._cosine_similarity(Counter(), Counter({"a": 1})) == 0.0
    assert abs(SchemaManager._cosine_similarity(Counter({"a": 2}), Counter({"a": 5})) - 1.0) < 1e-9
