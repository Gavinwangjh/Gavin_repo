# docker/Dockerfile
FROM python:3.12-slim

# System dependencies needed for psycopg2, pgvector, python client, and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Create a non-root user to run the application
# -m: create home directory (/home/appuser)
# -s: set default shell to bash (useful for docker exec)
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install uv \
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files separately for better caching
# If code changes and dependencies don't, Docker can reuse the cached layer
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# Change ownership of the application files to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Path for direct access python/cli without "uv run" prefix
ENV PATH="/app/.venv/bin:$PATH"

# Env variables for the application
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "python", "-m", "main"]