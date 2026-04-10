 This guide explains the tools used in this project, the project structure, and how to write and run code.

---

## Project structure

```text
project/
├── config/
│   ├── logger.py       # logging configuration — do not modify
│   └── settings.py     # reads environment variables from .env and other configuration
│
├── src/
│   ├── api/            # FastAPI endpoints and route definitions
│   ├── core/           # shared utilities and base exceptions
│   ├── db/
│   │   ├── base.py     # SQLAlchemy base class
│   │   ├── connection.py # Database engine and session management
│   │   └── models/     # database table definitions (SQLAlchemy)
│   │
│   └── temporal/       # All Temporal-related logic
│       ├── activities/ # individual atomic tasks (e.g., API calls, DB writes)
│       ├── workflows/  # orchestration logic (calling activities in order)
│       ├── client.py   # centralized Temporal client connection helper
│       └── worker.py   # worker-specific setup and registration
│
├── alembic/            # database migration files
│   └── versions/       # auto-generated migration scripts
│
├── docker/             # Docker configuration and environment setup
│   ├── docker-compose.yml
│   └── Dockerfile
│
├── scripts/            # standalone Python scripts that are used to interact with the system manually (e.g., triggering workflows, running one-off tasks, maintenance scripts, testing helpers)
│
├── main.py             # application entry point — starts the Temporal worker
├── pyproject.toml      # project dependencies (uv) and tool configuration
├── Makefile            # shortcuts for common commands (make setup, make start)
└── .env                # local configuration and secrets (never committed to Git)
```

---

## How code execution works

You write code on your computer. Docker runs it. The connection between them is a **volume mount** — a feature of Docker that makes a folder on your computer appear inside the container.

```
Your computer                Docker container
─────────────────            ─────────────────────
~/project/src/    ←────────→  /app/src/
~/project/main.py ←────────→  /app/main.py
```

When you save a file, the container immediately sees the new version. You do not need to rebuild Docker or restart anything for most changes. For changes to dependencies (`pyproject.toml`), you do need to rebuild: `make build`.

---

## Tools explained

### uv — dependency manager

`uv` manages the Python packages your project needs. It reads `pyproject.toml` and installs everything listed there into `.venv`.

Common commands:

```bash
# Install all dependencies (run after cloning or after pyproject.toml changes)
uv sync

# Add a new package to the project
uv add requests

# Add a package only for development (testing, linting — not needed in production)
uv add --dev pytest
```

After adding a package with `uv add`, it updates both `pyproject.toml` and `uv.lock`. Commit both files so your teammates get the same packages.

After adding a new package, rebuild the Docker image so the container also has it:

```bash
make build
make start
```

### SQLAlchemy — working with the database

SQLAlchemy is a library that lets you work with PostgreSQL from Python without writing raw SQL. You define tables as Python classes, and SQLAlchemy handles translating your operations into SQL.

A model (table definition) looks like this:

```python
# src/db/models/my_model.py
from sqlalchemy import Column, Integer, String, DateTime, func
from src.db.sandbox_base import SandboxBase

class Task(SandboxBase):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

To query the database in your code:

```python
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models.my_model import Task

# Insert a row
async def create_task(session: AsyncSession, name: str) -> Task:
    task = Task(name=name)
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task

# Query rows
async def get_tasks(session: AsyncSession) -> list[Task]:
    result = await session.execute(select(Task))
    return result.scalars().all()
```

The database connection is already configured in `src/db/connection.py`. You do not need to set it up.

**Before creating new models, discuss with the project owner.** New models must be approved before being pushed.

### Alembic — database migrations

When you add or change a model, the database does not automatically update. You need to create a **migration** — a file that describes what changed and how to apply it.

Think of migrations as a version history for your database schema. Each migration is a numbered file in `alembic/versions/`.

**Create a migration after changing a model:**

```bash
make migrate-create MSG="add tasks table"
```

This generates a new file in `alembic/versions/`. Open it and verify the generated changes look correct before applying.

**Apply migrations (update the database):**

```bash
make migrate
```

**Roll back the last migration:**

```bash
make migrate-down
```

**See migration history:**

```bash
make migrate-history
```

Important: always run `make migrate` after pulling new code from the repository, in case your teammates added new migrations.

### structlog — logging

The project uses `structlog` for logging. Unlike Python's built-in `logging`, structlog produces structured output — each log line is a JSON object with named fields. This makes logs much easier to read, search, and process.

Do not configure logging yourself. Import and use the pre-configured logger:

```python
import structlog

log = structlog.get_logger(__name__)

# Usage
log.info("task_started", task_id=42, name="process_file")
log.warning("file_not_found", path="/tmp/data.csv")
log.error("database_error", exc_info=True, table="tasks")
```

Each call accepts the event name as the first argument, followed by any number of keyword arguments that appear as fields in the output:

```json
{"event": "task_started", "level": "info", "task_id": 42, "name": "process_file", "timestamp": "..."}
```

Avoid `print()` for debugging. Use `log.debug()` instead — it can be turned off in production without changing code.

---

## Writing code

### Where to put new code

| What you are building | Where it goes |
|---|---|
| A workflow (orchestration logic) | `src/workflows/` |
| An activity (a single step of work) | `src/workflows/` or `src/activities/` |
| Reusable helpers, utilities | `src/core/` |
| Database models | `src/db/models/` (requires approval) |
| A new feature with multiple files | `src/<feature_name>/` |

### Principle: one file, one responsibility

Do not put unrelated code in the same file. If a file grows too large or handles more than one concern, split it.

Good structure for a scraping feature:

```text
src/scraping/
├── scraper.py      # fetches raw data from URLs
├── parser.py       # extracts structured data from raw content
└── storage.py      # saves results to the database
```

Bad:

```text
src/utils.py        # scraping + parsing + database + formatting all in one file
```

### Using the logger in your code

```python
# At the top of every file that needs logging
import structlog

log = structlog.get_logger(__name__)
```

`__name__` automatically sets the logger name to the module path (e.g., `src.scraping.scraper`), which makes it easy to see in logs which file produced which message.

### Using settings

Environment variables are available through the settings object:

```python
from config.settings import settings

print(settings.temporal_host)
print(settings.DATABASE_URL)
```

Do not read `os.environ` directly. Always use `settings`.

---

## Running tests

```bash
# Run all tests with a coverage report
make test

# Run tests without coverage (faster, useful during development)
make test-fast
```

Tests go in the `tests/` folder. Mirror the structure of `src/`:

```text
tests/
└── workflows/
    └── test_example_workflow.py
```

---

## Code quality

### Linting

Checks your code for errors, style issues, and common mistakes:

```bash
make lint
```

### Formatting

Automatically fixes formatting:

```bash
make format
```

Both `lint` and `format` use `ruff`, which is configured in `pyproject.toml`. You do not need to configure it.

Pre-commit hooks run `lint` automatically before every `git commit`. If your code has errors, the commit will be rejected until you fix them. This is intentional.

---

## Accessing the database directly

To open a PostgreSQL shell inside the running container:

```bash
make shell-db
```

This drops you into `psql` where you can run SQL queries directly. Useful for debugging.

To open a shell inside the app container:

```bash
make shell-app
```

---

## Summary

1. Write code in `src/`, organized by feature
2. Use `uv add <package>` to add dependencies, then `make build`
3. After changing models, run `make migrate-create` then `make migrate`
4. Use `structlog` for logging, `settings` for configuration
5. Run `make lint` and `make test` before pushing code