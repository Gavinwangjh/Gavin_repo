# ETL Prototype Operations Guide

This file records the prototype-specific operating notes for the organisation ETL pipeline.
For the short project overview, use [README.md](README.md).

## Verified Status

The following path has been tested end to end:

```text
extract_organisations
discover_websites
enrich_websites
enrich_sustainability
transform_organisations
load_organisations
```

Verified command:

```powershell
docker compose --env-file .env exec -T app uv run python src/temporal/scripts/run_etl.py --source au_wgea --limit 1
```

Observed successful result:

```text
status: successful
received: 1
```

Records can be skipped when the registration number already exists in the database. That is expected duplicate-protection behavior.

## Services

| Service | Purpose | URL / Port |
|---|---|---|
| PostgreSQL | Stores organisation records | localhost:5433 |
| Temporal | Runs ETL workflows | localhost:7233 |
| Temporal UI | Shows workflow history | http://localhost:8080 |
| Streamlit frontend | Views records and triggers workflows | http://localhost:8501 |

## Start And Stop

Start or rebuild all services:

```powershell
docker compose --env-file .env up -d --build
```

Check containers:

```powershell
docker compose --env-file .env ps
```

Stop containers while keeping database data:

```powershell
docker compose --env-file .env down
```

Delete local database data and start fresh:

```powershell
docker compose --env-file .env down -v
docker compose --env-file .env up -d --build
docker compose --env-file .env exec -T app uv run alembic upgrade head
```

## Database

Run migrations:

```powershell
docker compose --env-file .env exec -T app uv run alembic upgrade head
```

Open a database shell:

```powershell
docker compose --env-file .env exec db psql -U sandbox -d sandbox_db
```

Useful SQL:

```sql
SELECT COUNT(*) FROM "Organisations";

SELECT
  "OrganisationId",
  "OrganisationName",
  "OrganisationRegistrationNumber",
  "SourceName"
FROM "Organisations"
ORDER BY "OrganisationId" DESC
LIMIT 20;
```

## Data Sources

| Source key | Source | Status |
|---|---|---|
| `au` | data.gov.au organisation data | Implemented |
| `au_wgea` | WGEA public dataset | Verified end to end |
| `uk` | UK Companies House | Implemented |
| `us_sec` | SEC EDGAR | Implemented, but SEC may return `403 Forbidden` |

## Known Limitations

- SEC EDGAR may reject requests with `403 Forbidden`.
- UK SIC-to-ANZSIC mapping is not implemented.
- Website discovery uses domain guessing and page validation, so some organisations will not match.
- Duplicate handling currently skips existing registration numbers instead of updating missing fields.
- Public source quality varies by dataset.

## Recommended Next Improvements

- Add source-specific retry and backoff policies.
- Add upsert/merge behavior for duplicate registration numbers.
- Implement SIC-to-ANZSIC mapping for UK records.
- Add pagination/resume tracking for larger ingestion runs.
- Add more activity-level tests for transform and load behavior.
