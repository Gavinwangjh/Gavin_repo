# Makefile - in the root of the project, to build the project and run tests
#
# Use: make <target>
# Command list: make help

.PHONY: help setup start stop restart build ps logs logs-app logs-temporal \
        migrate migrate-down migrate-create migrate-history \
        test test-fast lint format shell-app shell-db clean reset
# Paths
ROOT_DIR := $(shell pwd)
ENV_FILE := .env
DOCKER_DIR := docker
DOCKER_ENV := $(DOCKER_DIR)/.env

COMPOSE = docker compose -f docker/docker-compose.yml --env-file $(shell pwd)/.env
APP_SERVICE = app
DB_SERVICE = postgres

# ─── Setup ───────────────────────────────────────────────────

setup:          ## Initial setup — run once after cloning
	@echo "Checking dependencies..."
	@command -v uv >/dev/null 2>&1 || \
		(echo "ERROR: uv is not installed. Install it: https://docs.astral.sh/uv/getting-started/installation/"; exit 1)
	@command -v docker >/dev/null 2>&1 || \
		(echo "ERROR: Docker is not installed."; exit 1)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ""; \
		echo "=== INITIAL PROJECT SETUP ==="; \
		echo ""; \
		read -p "  Project name (e.g. unisa-07): " pname; \
		while [ -z "$$pname" ]; do \
			echo "  Project name cannot be empty."; \
			read -p "  Project name (e.g. unisa-07): " pname; \
		done; \
		read -p "  External Postgres port [5432]: " pport; \
		pport=$${pport:-5432}; \
		read -p "  External Temporal UI port [8080]: " uiport; \
		uiport=$${uiport:-8080}; \
		db_pass=$$(openssl rand -hex 8); \
		sed -i.bak \
			-e "s|^PROJECT_NAME=.*|PROJECT_NAME=$$pname|" \
			-e "s|^DB_PORT=.*|DB_PORT=$$pport|" \
			-e "s|^TEMPORAL_UI_PORT=.*|TEMPORAL_UI_PORT=$$uiport|" \
			-e "s|^DB_PASSWORD=.*|DB_PASSWORD=$$db_pass|" \
			-e "s|@sandbox:$$pport|@sandbox:5432|" \
			.env && rm -f .env.bak; \
		echo ""; \
		echo "  Project : $$pname"; \
		echo "  DB port : $$pport"; \
		echo "  UI port : $$uiport"; \
		echo "  DB pass : $$db_pass (saved to .env)"; \
		echo ""; \
		echo "Setup complete."; \
	else \
		echo ".env already exists — skipping interactive setup."; \
	fi
	@echo "Installing Python dependencies..."
	@uv sync
	@echo "Installing pre-commit hooks..."
	@uv run pre-commit install
	@uv run pre-commit install --hook-type pre-push
	@ln -sf ../$(ENV_FILE) $(DOCKER_ENV)
	@echo ""
	@echo "Done. Next step: make start"


# ─── Lifecycle ───────────────────────────────────────

start:          ## Start all services
	@docker info >/dev/null 2>&1 || \
		(echo "ERROR: Docker is not running. Start Docker Desktop first."; exit 1)
	@test -f .env || \
		(echo "ERROR: .env file not found. Run: make setup"; exit 1)
	$(COMPOSE) up -d
	@echo ""
	@echo "  Temporal UI -> http://localhost:$$(grep TEMPORAL_UI_PORT .env | cut -d= -f2)"
	@echo "  Postgres    -> localhost:$$(grep ^POSTGRES_PORT .env | cut -d= -f2)"
	@echo "  Logs        -> make logs-app"


stop:           ## Stop all services (data is preserved)
	$(COMPOSE) down

restart:        ## Restart all services (stop + start)
	$(COMPOSE) restart

build:          ## Rebuild app Docker image (run after adding new packages)
	$(COMPOSE) build $(APP_SERVICE)

ps:             ## Status of the running containers
	$(COMPOSE) ps

logs:           ## Stream logs of all services
	$(COMPOSE) logs -f

logs-app:       ## Stream logs of the app service only
	$(COMPOSE) logs -f $(APP_SERVICE)

logs-temporal:  ## Stream logs of the temporal service only
	$(COMPOSE) logs -f temporal

# ─── Database ────────────────────────────────────────────────

migrate:        ## Apply all pending migrations
	$(COMPOSE) exec $(APP_SERVICE) uv run alembic upgrade head

migrate-down:   ## Roll back last migration
	$(COMPOSE) exec $(APP_SERVICE) uv run alembic downgrade -1

migrate-create: ## Create new migration: make migrate-create MSG="add tasks table"
	$(COMPOSE) exec $(APP_SERVICE) uv run alembic revision --autogenerate -m "$(MSG)"

migrate-history:## Get the history of migrations (alembic history --verbose)
	$(COMPOSE) exec $(APP_SERVICE) uv run alembic history --verbose

# ─── Testing & quality ───────────────────────────────────────

test:           ## Run all tests with coverage report
	uv run pytest --cov=src --cov-report=term-missing -v

test-fast:      ## Run tests without coverage
	uv run pytest -x -q

lint:           ## Check code quality
	uv run pre-commit run --all-files
	uv run pyright src/

format:         ## Fix code formatting
	uv run ruff format .
	uv run ruff check --fix .

# ─── Shells ──────────────────────────────────────────────────

shell-app:      ## Open shell inside app container
	$(COMPOSE) exec $(APP_SERVICE) /bin/bash

shell-db:       ## Open psql shell inside database container
	$(COMPOSE) exec $(DB_SERVICE) psql -U $${POSTGRES_USER:-sandbox} -d $${POSTGRES_DB:-sandbox_db}

# ─── Cleanup ─────────────────────────────────────────────────

clean:          ## Remove containers and local images
	$(COMPOSE) down --rmi local

reset:          ## ⚠ REMOVE ALL DATA - delete all containers and data
	@echo "⚠ This will remove ALL from the database. Continue? [y/N]"
	@read ans && [ $${ans:-N} = y ] || exit 1
	$(COMPOSE) down -v --remove-orphans
	@echo "✓ Reset completed. Run: make start && make migrate"