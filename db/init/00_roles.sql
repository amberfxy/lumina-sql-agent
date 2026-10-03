-- Least-privilege role used by the API. The development password is overridden in
-- real deployments (see docs/security.md); grants are applied in 03_grants.sql after
-- the tables exist.
CREATE ROLE lumina_reader LOGIN PASSWORD 'lumina_reader'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

ALTER ROLE lumina_reader SET default_transaction_read_only = on;
ALTER ROLE lumina_reader SET statement_timeout = '15s';
ALTER ROLE lumina_reader SET idle_in_transaction_session_timeout = '30s';
