-- The reader can connect, see the application schema, and SELECT from its tables. Nothing else:
-- no writes, no sequences, no CREATE, and no membership in pg_read_server_files or
-- pg_execute_server_program (so pg_read_file() and COPY ... TO PROGRAM are denied).
REVOKE ALL ON DATABASE lumina FROM PUBLIC;
GRANT CONNECT ON DATABASE lumina TO lumina_reader;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO lumina_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO lumina_reader;
