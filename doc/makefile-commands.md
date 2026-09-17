# Makefile Commands

The `Makefile` wraps common Docker, migration, and test commands.

## Setup

```bash
make setup
```

Creates `.env` from `.env.example`, installs Python dependencies with `uv sync`, and installs pre-commit hooks.

## Start And Stop

```bash
make start
make stop
make restart
make ps
```

- `make start` starts all Docker services.
- `make stop` stops services while preserving database data.
- `make restart` restarts running services.
- `make ps` shows container status.

After startup:

- Temporal UI: http://localhost:8080
- Frontend: http://localhost:8501
- PostgreSQL: `localhost:5433`

## Build

```bash
make build
```

Rebuilds the app image. Run this after dependency or Dockerfile changes.

## Logs

```bash
make logs
make logs-app
make logs-temporal
```

Use app logs to inspect worker activity and Temporal logs to inspect server startup or workflow orchestration issues.

## Database

```bash
make migrate
make migrate-down
make migrate-history
make migrate-create MSG="describe change"
```

Run `make migrate` after the first startup and whenever migrations change.

## Testing And Quality

```bash
make test
make test-fast
make lint
make format
```

The focused commands used during validation were:

```bash
uv run ruff check src tests alembic
uv run pytest -p no:cacheprovider tests
```

## Shells

```bash
make shell-app
make shell-db
```

`make shell-app` opens a shell inside the app worker container.
`make shell-db` opens a PostgreSQL shell for `sandbox_db`.

## Cleanup

```bash
make clean
make reset
```

- `make clean` removes stopped containers and local images.
- `make reset` removes containers, networks, and database data.

Use `make reset` only when you intentionally want a fresh local database.
