All common tasks in this project are available as `make` commands. This avoids having to remember long Docker or Python commands.

Run `make help` to see the list of commands directly in your terminal.

---

## Setup and lifecycle

### `make setup`

Run this once after cloning the repository.

It does the following:
1. Copies `.env.example` to `.env` (only if `.env` does not already exist)
2. Installs Python dependencies locally via `uv sync`
3. Installs pre-commit hooks

### `make start`

Starts all Docker containers in the background:
- PostgreSQL database
- Temporal server
- Temporal UI
- Python worker (your application)

After running this, open [http://localhost:8080](http://localhost:8080) to verify Temporal is up.

### `make stop`

Stops all containers. Your data is preserved — the database volume is not deleted.

### `make restart`

Stops and starts all containers. Use this after modifying `main.py` to reload the worker.

### `make build`

Rebuilds the Docker image for the Python application. Run this after:
- Adding a new package with `uv add`
- Changing the `Dockerfile`

### `make ps`

Shows the status of all containers. Useful for checking whether everything started correctly.

```
NAME                    STATUS
sandbox_postgres        running (healthy)
sandbox_temporal        running
sandbox_temporal_ui     running
datacollection_python   running
```

### `make logs`

Streams logs from all containers. Press `Ctrl+C` to stop.

### `make logs-app`

Streams logs from the Python application only. Use this to see what your code is doing.

### `make logs-temporal`

Streams logs from the Temporal server. Useful if workflows are not starting.

---

## Database

### `make migrate`

Applies all pending database migrations. Run this:
- After the first `make start`
- After pulling new code (in case teammates added migrations)
- After running `make migrate-create` and verifying the generated file

### `make migrate-create MSG="your description"`

Creates a new migration file based on changes to your models.

Example:
```bash
make migrate-create MSG="add tasks table"
```

This generates a new file in `alembic/versions/`. Open it and verify the changes before applying with `make migrate`.

### `make migrate-down`

Rolls back the last applied migration. Useful if you applied a migration with an error.

### `make migrate-history`

Shows the list of all migrations and which ones have been applied.

---

## Testing and code quality

### `make test`

Runs all tests and generates a coverage report. The report shows which lines of code are covered by tests and which are not.

### `make test-fast`

Runs tests without coverage. Faster — use this during active development when you want quick feedback.

### `make lint`

Checks your code for errors, style issues, and type problems using `ruff` and `pyright`. Fix any reported issues before pushing.

### `make format`

Automatically fixes formatting issues using `ruff`. Run this before `make lint` if you want to fix things automatically.

---

## Shell access

### `make shell-app`

Opens a bash shell inside the running Python container. Useful for running Python scripts manually or inspecting the container.

```bash
make shell-app
# now you are inside the container
python main.py
uv run pytest
```

### `make shell-db`

Opens a PostgreSQL shell (`psql`) inside the database container. Useful for running SQL queries directly.

```bash
make shell-db
# now you are in psql
\dt                        -- list all tables
SELECT * FROM tasks;       -- query a table
\q                         -- quit
```

---

## Cleanup

### `make clean`

Removes stopped containers and locally built Docker images. Does not delete database data.

### `make reset`

Deletes everything — containers, networks, and the database volume. All data is lost.

This command asks for confirmation before proceeding. Use it when you want a completely fresh start.

After reset, run:
```bash
make start
make migrate
```