# ETL Data Pipeline System

This project is a Docker-based ETL data pipeline for collecting, enriching, transforming, and loading organisation data.

It uses:

- Python 3.12
- Temporal workflows and activities
- PostgreSQL with pgvector
- Alembic migrations
- Streamlit for a lightweight frontend
- Docker Compose for local orchestration

## Pipeline

The main Temporal workflow runs these steps:

1. Extract organisation records from a configured data source.
2. Discover likely organisation websites.
3. Enrich website/contact fields.
4. Discover sustainability pages and reports.
5. Transform records into the database ingestion format.
6. Load records into PostgreSQL with duplicate registration-number checks.

Supported sources:

- `au`
- `au_wgea`
- `uk`
- `us_sec`

Note: `us_sec` can return `403 Forbidden` from SEC endpoints depending on request policy, IP reputation, or SEC rate limiting. The rest of the pipeline has been verified with `au_wgea`.

## Quick Start

```bash
make start
make migrate
```

Open:

- Temporal UI: http://localhost:8080
- Frontend: http://localhost:8501
- PostgreSQL: `localhost:5433`

## Run An ETL Workflow

Run a small end-to-end workflow from inside the app container:

```bash
docker compose --env-file .env exec -T app uv run python src/temporal/scripts/run_etl.py --source au_wgea --limit 1
```

Expected successful output shape:

```text
{
  "received": 1,
  "inserted": 0,
  "skipped": 1,
  "status": "successful"
}
```

`inserted` may be `0` when the record already exists and is skipped by duplicate registration-number checks.

## Quality Checks

```bash
uv run ruff check src tests alembic
uv run pytest -p no:cacheprovider tests
```

Current focused tests cover the website discovery helpers and existing-website preservation behavior.

## Services

| Service | Address | Purpose |
|---|---|---|
| Temporal UI | http://localhost:8080 | Monitor workflows |
| Frontend | http://localhost:8501 | View and trigger ETL runs |
| PostgreSQL | localhost:5433 | Organisation database |
| App worker | Docker service `app` | Runs Temporal worker |
| Temporal server | localhost:7233 | Workflow orchestration |

## Documentation

Additional project notes are in:

- [Getting Started](doc/getting-started.md)
- [Development Guide](doc/development-guide.md)
- [Temporal Workflows](doc/temporal-workflows.md)
- [Makefile Commands](doc/makefile-commands.md)
- [Git Workflow](doc/git-workflow.md)
