# Temporal Workflows

Temporal runs the ETL pipeline as a durable workflow. If an activity fails, Temporal records the failure and can retry according to its policy.

## Current Workflow

The production ETL workflow is `ETLWorkflow` in `src/temporal/workflows/etl_workflow.py`.

```text
extract_organisations
discover_websites
enrich_websites
enrich_sustainability
transform_organisations
load_organisations
```

The worker listens on:

```text
sandbox-task-queue
```

This queue name is used consistently by:

- `main.py`
- `src/temporal/worker.py`
- `src/temporal/scripts/run_etl.py`

## Activities

Activities live in `src/temporal/activities/`.

| Activity | Purpose |
|---|---|
| `extract_organisations` | Fetches source records from AU, WGEA, UK, or SEC sources |
| `discover_websites` | Guesses and validates likely organisation website URLs |
| `enrich_websites` | Adds website/contact details where available |
| `enrich_sustainability` | Looks for sustainability and report URLs |
| `transform_organisations` | Maps source records into a common load format |
| `load_organisations` | Inserts records into PostgreSQL or skips duplicates |

## Running A Workflow

From the repository root:

```bash
docker compose --env-file .env exec -T app uv run python src/temporal/scripts/run_etl.py --source au_wgea --limit 1
```

Open Temporal UI:

```text
http://localhost:8080
```

## Workflow Rules

- Workflow code should orchestrate activities only.
- Network calls and database writes belong in activities.
- Activity imports inside workflow files must remain inside `workflow.unsafe.imports_passed_through()`.
- Activity inputs and outputs should be JSON-serialisable dictionaries or simple values.

## Adding A New Source

1. Add extraction logic to `extract_organisations`.
2. Add any source-specific normalisation in `transform_organisations`.
3. Confirm the source key is accepted by `src/temporal/scripts/run_etl.py`.
4. Run a small workflow with `--limit 1`.
5. Check Temporal UI and database load results.

## Troubleshooting

If a workflow stays running:

- Check `docker compose --env-file .env logs app`.
- Confirm the worker is listening on `sandbox-task-queue`.
- Confirm the activity is registered in `main.py` or `src/temporal/worker.py`.

If an external source fails:

- Check whether the source returned 403, 404, timeout, or invalid data.
- Re-run with a small limit.
- Keep source-specific failures separate from pipeline failures.
