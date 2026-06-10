# Organisation ETL Prototype

This repository contains a prototype ETL pipeline for discovering, enriching, transforming and loading organisation data from public sources into PostgreSQL.

The prototype uses:

* Python
* Temporal workflows
* PostgreSQL
* Docker Compose
* Streamlit frontend

---

## Current branch

Use the `prototype` branch:

```powershell
git switch prototype
git pull origin prototype
```

---

## Main services

| Service            | Purpose                                   | URL / Port                                |
| ------------------ | ----------------------------------------- | ----------------------------------------- |
| PostgreSQL         | Stores final organisation records         | localhost:5433 or `.env` external DB port |
| Temporal           | Runs ETL workflows                        | 7233                                      |
| Temporal UI        | View workflow runs and activity results   | http://localhost:8080                     |
| Streamlit frontend | View loaded records and trigger workflows | http://localhost:8501                     |

---

## Start the project

From the repository root:

```powershell
docker compose --env-file .env up -d --build (builds docker containers)
```

Check containers:

```powershell
docker compose --env-file .env ps
```

Stop containers but keep database data:

```powershell
docker compose --env-file .env down
```

Stop containers and delete local database data:

```powershell
docker compose --env-file .env down -v
```

Only use `down -v` when you want a fresh database.

---

## Fresh database setup

The database schema is created from:

```text
init-scripts/create_schema.sql
```

This creates the required tables:

* `Countries`
* `PartnerType`
* `OrganisationSize`
* `CategoryTypes`
* `Organisations`

PostgreSQL only runs files in `init-scripts/` when the database volume is created for the first time.

To force a clean setup:

```powershell
docker compose --env-file .env down -v
docker compose --env-file .env up -d --build
```

Check tables:

```powershell
docker compose --env-file .env exec db psql -U <DB_USER> -d <DB_NAME> -c "\dt" (goes into sql database)
```

Expected output should include:

```text
CategoryTypes
Countries
OrganisationSize
PartnerType
Organisations (Main Databse)
```

---

## Use the frontend

Open:

```text
http://localhost:8501
```

The frontend can:

* View loaded organisation records
* Filter by country, source and category
* Search by organisation name or registration number
* Trigger ETL workflows using source and limit options

Temporal UI:

```text
http://localhost:8080
```

Temporal UI gives summaries of the workflow, shows skipped entries, has auto retry capabilities if the pipeline break at a certain step while also having a switch to cancel running workflows if required. 

---

## Run ETL workflows manually

The workflows can also be triggered manually from the root using this command:

```powershell
docker compose --env-file .env exec app uv run python src/temporal/scripts/run_etl.py --source {Source_name} --limit {respective_limits}
```

Loading time for sources may vary. As an estimate, the UK source loads 100 entries in roughly 10 mins. Majority time is spent on discovery and enrichment.

The AU sources take about 5 minutes per run for 100 entries on average to run as of now. 

---

## Check loaded records

Open PostgreSQL:

```powershell
docker compose --env-file .env exec db psql -U <DB_USER> -d <DB_NAME>
```

Count loaded organisations:

```sql
SELECT COUNT(*) FROM "Organisations";
```

View records:

```sql
SELECT
  "OrganisationId",
  "OrganisationName",
  "OrganisationRegistrationNumber",
  "SourceName"
FROM "Organisations"
ORDER BY "OrganisationId";
```

Clear only organisation records:

```sql
DELETE FROM "Organisations";
```

Exit PostgreSQL:

```sql
\q
```

---

## Supported sources

| Source key | Source                        | Notes                                                      |
| ---------- | ----------------------------- | ---------------------------------------------------------- |
| `au`       | data.gov.au organisation data | Provides ABN and organisation/company names                |
| `au_wgea`  | WGEA public dataset           | Provides ABN, ANZSIC, division and organisation size range |
| `uk`       | UK Companies House            | Provides company number and SIC codes                      |
| `us_sec`   | SEC EDGAR                     | Attempted but blocked by 403 Forbidden                     |

---

## Current ETL flow

```text
extract_organisations
    ↓
discover_websites
    ↓
enrich_websites
    ↓
enrich_sustainability
    ↓
transform_organisations
    ↓
load_organisations
```

Main files:

```text
src/temporal/workflows/etl_workflow.py
src/temporal/activities/extract_organisations.py
src/temporal/activities/discover_websites.py
src/temporal/activities/enrich_websites.py
src/temporal/activities/enrich_sustainability.py
src/temporal/activities/transform_organisations.py
src/temporal/activities/load_organisations.py
src/temporal/scripts/run_etl.py
src/frontend/app.py
```

---

## What transformation does

The transformation stage converts source-specific records into one common database-ready structure.

It handles:

* Standardising fields from different public sources
* Mapping ABNs/company numbers to `OrganisationRegistrationNumber`
* Mapping country values to internal country codes
* Mapping WGEA ANZSIC divisions to internal category codes
* Preserving SIC/ANZSIC source codes and descriptions
* Keeping organisation size conservative when source data is unclear
* Preparing records for validation and loading

Important limitation:

UK Companies House provides SIC codes, but the client structure is more aligned with ANZSIC. SIC-to-ANZSIC mapping is not currently implemented.

---

## Common errors and fixes

### 1. Frontend error: `relation "Organisations" does not exist`

Meaning:

This error shows up when the database tables were not initialised/made. The create_schema.sql file builds the required tables at the first start of the containers.

Fix:

```powershell
docker compose --env-file .env down -v
docker compose --env-file .env up -d --build
```

Then check:

```powershell
docker compose --env-file .env exec db psql -U sandbox -d sandbox_db -c "\dt"
```

If tables are still missing, check that this file exists:

```text
init-scripts/create_schema.sql
```

---

### 2. `Did not find any relations`

Meaning:

PostgreSQL is running, but the database has no tables.

Fix:

Make sure `init-scripts/create_schema.sql` exists, then recreate the database volume:

```powershell
docker compose --env-file .env down -v
docker compose --env-file .env up -d --build
```

---

### 3. Frontend error: password authentication failed

Meaning:

The frontend is using the wrong database username/password.

Fix:

The frontend should use the same database URL as the app service:

```yaml
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@db:5432/${DB_NAME}
```

Do not hardcode database credentials in `docker-compose.yml`.

---

### 4. `uv is not recognized`

Meaning:

`uv` is not installed on the local Windows machine.

Fix:

Run `uv` inside the Docker app container instead:

```powershell
docker compose --env-file .env exec app uv --version
```

To add dependencies:

```powershell
docker compose --env-file .env exec app uv add streamlit pandas
```

---

### 5. Docker error: `TLS handshake timeout`

Meaning:

Docker could not reach Docker Hub to download an image.

Fix:

Try:

```powershell
docker pull python:3.12-slim
```

If it fails:

* Restart Docker Desktop
* Check internet/VPN
* Retry the build

```powershell
docker compose --env-file .env up -d --build
```

---

### 6. Workflow inserted fewer records than requested

Example:

```json
{
  "received": 25,
  "inserted": 11,
  "skipped": 14
}
```

Meaning:

The workflow received 25 transformed records, but some were skipped during loading.

Common reasons:

* Missing organisation name
* Duplicate registration number already exists

Check Temporal activity result for skipped record details.

---

### 7. Frontend does not show latest code changes

Meaning:

The frontend container may not be using the mounted source code.

Fix:

Ensure the `frontend` service has:

```yaml
volumes:
  - .:/app
  - /app/.venv
working_dir: /app
```

Restart frontend:

```powershell
docker compose --env-file .env restart frontend
```

Hard refresh browser:

```text
Ctrl + F5
```

---

## Useful commands

Show running containers:

```powershell
docker compose --env-file .env ps
```

View app logs:

```powershell
docker compose --env-file .env logs -f app
```

View frontend logs:

```powershell
docker compose --env-file .env logs -f frontend
```

Restart app worker:

```powershell
docker compose --env-file .env restart app
```

Restart frontend:

```powershell
docker compose --env-file .env restart frontend
```

Open database shell:

```powershell
docker compose --env-file .env exec db psql -U sandbox -d sandbox_db
```

List tables (when inside database shell):

```sql
\dt
```

Count organisation records:

```sql
SELECT COUNT(*) FROM "Organisations";
```

Clear organisation records:

```sql
DELETE FROM "Organisations";
```

---

## Known limitations

* UK SIC-to-ANZSIC mapping is not implemented.
* UK "organisation name" needs specialised cleaning to get clear organisation names.
* Organisation size is not always reliable from public sources.
* WGEA provides broad size ranges that do not always map cleanly to Small/Medium/Large.
* PDF text extraction for employee counts is not implemented.
* Duplicate handling currently skips existing registration numbers instead of updating missing fields.
* Public source quality varies by country and dataset.
* SEC EDGAR source was attempted but blocked by 403 Forbidden.

---

## Future improvements

Recommended next improvements:

* Implement SIC-to-ANZSIC mapping for UK records.
* Add upsert/merge logic instead of only skipping duplicates.
* Automatic new batch ingestion to be implemented.
* Extract employee count from sustainability/annual report PDFs.
* Extractor supports offset/pagination/ resume tracking(remembering where it stopped last time).
* Loader uses database unique constraint.
* Improve source metadata fields such as source country/type/display name.
* Add more sources for countries such as India, Canada and Ireland (recommendations included in the handover document).
* Improve frontend with workflow status and log display.

---

## Version control

Work is pushed to the `prototype` branch currently.


