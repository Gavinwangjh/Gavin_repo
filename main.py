"""
Main entry point for the Datacollection Python app.

- Registers Temporal workflows and activities.
- Starts the worker that listens to the task queue.
- Blocking call: worker.run() keeps the process alive.

Example usage:
    uv run main.py
"""

import asyncio

import structlog
from temporalio.worker import Worker

from config.logger import setup_logging
from src.temporal.activities.example_activity import example_activity
from src.temporal.client import get_temporal_client
from src.temporal.workflows.example_workflow import ExampleWorkflow

setup_logging()
log = structlog.get_logger(__name__)


async def main():
    # 1. Add a shared Temporal client configuration
    client = await get_temporal_client()

    # 2. Worker initialization: register workflows and activities
    worker = Worker(
        client,
        task_queue="sandbox-task-queue",
        workflows=[ExampleWorkflow],
        activities=[example_activity],
    )

    log.info("Worker started", task_queue="sandbox-task-queue")

    # 3. Run the worker: this is a blocking call that keeps the process alive and listening for tasks
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Worker stopped by user")
