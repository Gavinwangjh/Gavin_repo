A development sandbox for building data collection pipelines using Python, Temporal, and PostgreSQL — all running inside Docker.

---

## What is this project?

This is a starter template. Everything you need to run a full development environment is already configured. You do not need to install Python, PostgreSQL, or any server software on your computer. Docker handles all of that.

Your job is to write Python code. The system runs it.

---

## Quick Start

```bash
git clone <repo-url>
cd datacollection-starter-kit-template

make setup
make start
make migrate
```

After that, open [http://localhost:8080](http://localhost:8080) to see the Temporal workflow dashboard.

---

## Documentation

Read these guides in order if you are new to the project:

| Guide                                            | What it covers |
|--------------------------------------------------|---|
| [Getting Started](docs/getting-started.md)       | Installation, setup, how to run the system |
| [Development Guide](docs/development-guide.md)   | How to write code, project structure, tools |
| [Temporal Workflows](docs/temporal-workflows.md) | What Temporal is, how to write workflows and activities |
| [Makefile Commands](docs/makefile-commands.md)   | All available `make` commands explained |
| [Git Workflow](docs/git-workflow.md)             | How to work with branches and pull requests |

---

## Services

| Service | Address | Purpose |
|---|---|---|
| Temporal UI | http://localhost:8080 | Monitor workflows |
| PostgreSQL | localhost:5432 | Database |
| App (worker) | — | Runs in background, listens for tasks |
