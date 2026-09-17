# Development Guide

This guide explains the structure of the ETL data pipeline and the usual development workflow.

## Project Structure

```text
config/                         Application settings and logging
src/frontend/                   Streamlit frontend
src/temporal/activities/        ETL activity implementations
src/temporal/workflows/         Temporal workflow orchestration
src/temporal/scripts/           Manual workflow trigger scripts
src/db/                         Database connection and model definitions
alembic/versions/               Database migrations
init-scripts/                   Initial PostgreSQL bootstrap SQL
tests/                          Unit tests
```

## Main ETL Flow

The workflow is defined in `src/temporal/workflows/etl_workflow.py`.

```text
extract_organisations
discover_websites
enrich_websites
enrich_sustainability
transform_organisations
load_organisations
```

Each activity should accept and return plain serialisable dictionaries because Temporal persists activity inputs and outputs.

## Running Locally

Start services:

```bash
make start
```

Apply migrations:

```bash
make migrate
```

Run a small workflow:

```bash
docker compose --env-file .env exec -T app uv run python src/temporal/scripts/run_etl.py --source au_wgea --limit 1
```

## Working With Activities

Keep activities focused:

- extraction code should only fetch and normalise source records
- enrichment code should add fields without changing unrelated values
- transform code should map source records into database-ready records
- load code should validate database references and insert or skip records

When adding an activity:

1. Add the function under `src/temporal/activities/`.
2. Register it in `src/temporal/worker.py`.
3. Add it to the workflow in `src/temporal/workflows/etl_workflow.py`.
4. Add focused tests for helper logic where possible.

## Database Changes

Database schema changes should be represented as Alembic migrations.

```bash
make migrate-create MSG="describe change"
make migrate
```

Always inspect generated migrations before applying them. The current ETL loader expects these tables:

- `Countries`
- `PartnerType`
- `OrganisationSize`
- `CategoryTypes`
- `Organisations`

## Testing

Run the fast test suite:

```bash
uv run pytest -p no:cacheprovider tests
```

Run lint checks:

```bash
uv run ruff check src tests alembic
```

Format changed Python files:

```bash
uv run ruff format src tests alembic
```

## Logging

Use `structlog` for runtime logs:

```python
import structlog

log = structlog.get_logger(__name__)
log.info("activity_started", source="au_wgea")
```

Avoid debug `print()` calls in long-lived worker code. Temporary prints in prototype extraction scripts should be removed or converted to structured logs before production use.

## Known External Constraints

Some public data sources can block or throttle requests. `us_sec` may return `403 Forbidden` from SEC endpoints even when the pipeline itself is healthy. Treat these as source-specific availability issues and keep the rest of the workflow resilient.
