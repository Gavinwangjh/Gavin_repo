This guide walks you through setting up and running the project from scratch. No prior experience with Docker, databases, or workflow systems is required — everything is explained.

---

## 1. Understanding the setup

Before installing anything, it helps to understand what this project actually does when you run it.

### What is Docker?

Docker is a tool that lets you run software in isolated environments called **containers**. Think of a container as a lightweight virtual machine — it has its own operating system, installed software, and configuration, completely separate from your computer.

This matters because:

- You do not need to install PostgreSQL, Python servers, or Temporal on your PC
- Every person on the team runs the exact same environment
- "It works on my machine" problems disappear

When you run `make start`, Docker starts three containers:

| Container | What it is |
|---|---|
| `sandbox_postgres` | A PostgreSQL database server |
| `sandbox_temporal` | The Temporal workflow server |
| `sandbox_temporal_ui` | A web dashboard for monitoring workflows |
| `datacollection_python` | Your Python application (the worker) |

Your code runs inside `datacollection_python`. That container reads your files directly from your computer — so when you save a file in your editor, the container sees the change immediately.

### What is `uv`?

`uv` is a Python package manager — similar to `pip`, but significantly faster. It manages your project's dependencies (the libraries your code needs) and creates a virtual environment (`.venv`).

When you run `uv sync`, it reads `pyproject.toml` and installs everything listed there.

### What is a virtual environment?

A virtual environment is an isolated folder (`.venv`) that contains a specific version of Python and all the packages your project needs. It prevents conflicts between projects that need different versions of the same library.

In this project there are two virtual environments:

- One on your computer — used by your IDE (PyCharm, VSCode) for autocomplete and code checking
- One inside Docker — used to actually run the application

They contain the same packages but are built for different operating systems and should never be mixed.

---

## 2. Prerequisites

### Docker

Install Docker Desktop. It includes everything you need.

- Mac/Windows: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- Linux:

```bash
sudo apt install docker.io docker-compose-plugin
sudo usermod -aG docker $USER
```

After the Linux command, **log out and log back in** for the group change to take effect. Then verify Docker works:

```bash
docker run hello-world
```

### Make

`make` is a command-line tool that runs shortcuts defined in `Makefile`. Instead of typing long Docker commands, you type `make start`.

- Mac: already installed (comes with Xcode command line tools)
- Linux: `sudo apt install make`
- Windows: install via [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and then `sudo apt install make`

### uv

`uv` is needed for local Python environment setup (your IDE needs it).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

---

## 3. Clone the repository

```bash
git clone <repo-url>
cd datacollection-starter-kit-template
```

---

## 4. Initial setup

Run this once after cloning:

```bash
make setup
```

This command does three things:

1. Copies `.env.example` to `.env` — your local configuration file
2. Runs `uv sync` — installs Python packages into your local `.venv`
3. Installs pre-commit hooks — automatic code checks before each commit

After this you should see a `.venv` folder in the project root. This is your local Python environment for the IDE.

### Configure your IDE interpreter

PyCharm needs to know where Python is installed so it can provide autocomplete and type checking.

In PyCharm: `Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Existing > select .venv/bin/python`

---

## 5. Start the system

```bash
make start
```

Docker will download the required images on the first run (this takes a few minutes). After that, all containers start in the background.

You can verify everything is running:

```bash
make ps
```

You should see four containers with status `running` or `healthy`.

---

## 6. Run database migrations

```bash
make migrate
```

This creates all the tables in the PostgreSQL database. You need to run this once after the first `make start`, and again whenever new migrations are added to the project.

What migrations are is explained in the [Development Guide](development-guide.md).

---

## 7. Verify everything works

Check the application logs:

```bash
make logs-app
```

You should see something like:

```
{"event": "Application connected to temporal", "level": "info", ...}
{"event": "Worker started, listening for tasks...", "level": "info", ...}
```

Open the Temporal UI in your browser:

```
http://localhost:8080
```

You should see the Temporal dashboard. If you see it, the system is fully operational.

---

## 8. Environment variables — what they are and why they matter

The `.env` file contains configuration values that the system needs to connect services together: database credentials, hostnames, ports.

This file is never committed to Git because it can contain sensitive information. That is why only `.env.example` exists in the repository — it is a template with safe default values.

Open `.env` and you will see entries like:

```bash
POSTGRES_USER=sandbox
POSTGRES_PASSWORD=sandbox
DATABASE_URL=postgresql+asyncpg://sandbox:sandbox@postgres:5432/sandbox_db
```

**Important — two different addresses for the database:**

| Context | Address to use |
|---|---|
| Code running inside Docker | `@postgres:5432` — the container name |
| Connecting from your PC (DBeaver, terminal) | `@localhost:5432` — exposed port |

Inside Docker, containers talk to each other using their service names (`postgres`, `temporal`). From your PC, you access them via `localhost` because Docker maps the ports.

You do not need to change anything in `.env` for the default setup to work.

---

## 9. Connecting to the database from your PC

If you want to inspect the database using a GUI tool like DBeaver or TablePlus:

| Field | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `sandbox_db` |
| Username | `sandbox` |
| Password | `sandbox` |

---

## 10. Common problems

### Docker permission denied (Linux)

```bash
sudo usermod -aG docker $USER
# log out and log back in
```

### App container keeps restarting

Check what error it reports:

```bash
make logs-app
```

The most common cause is a missing or broken `main.py`, or a configuration error in `.env`.

### Changes to my code are not reflected

Make sure:
- you saved the file in your editor
- the container is running (`make ps`)

If the worker does not automatically pick up the change, restart it:

```bash
make restart
```

### Port already in use

If port 5432 or 8080 is used by another application on your PC, edit `.env` and change `POSTGRES_PORT` or `TEMPORAL_UI_PORT` to a different value, then restart.

---

## Next steps

- Read the [Development Guide](development-guide.md) to learn how to write code and where to put it
- Read [Temporal Workflows](temporal-workflows.md) to understand how the workflow system works