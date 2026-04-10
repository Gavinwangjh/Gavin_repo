"""
This module defines the Temporal worker that listens for tasks
on the specified task queue and executes workflows and activities.
- The worker is responsible for executing workflows and activities
when they are scheduled by the Temporal server.
- It connects to the Temporal server using a shared client configuration.
- The worker registers the workflows and activities it can execute.
- To run the worker, execute this script. It will keep running and processing tasks until stopped.
Example usage:
    uv run src/temporal/worker.py
"""

import asyncio

import structlog
from temporalio.worker import Worker

from src.temporal.activities.example_activity import example_activity
from src.temporal.client import get_temporal_client

# Import workflows and activities to register them with the worker
from src.temporal.workflows.example_workflow import ExampleWorkflow

log = structlog.get_logger(__name__)


async def run_worker() -> None:
    # We use a shared client configuration to connect to the Temporal server.
    # This allows us to reuse the same connection across different parts of the application
    # (e.g., scripts, workers).
    client = await get_temporal_client()

    worker = Worker(
        client,
        task_queue="sandbox-task-queue",
        workflows=[ExampleWorkflow],
        activities=[example_activity],
    )

    log.info("Worker started", task_queue="sandbox-task-queue")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
