"""
Run this script with `temporal trigger run example_trigger.py` to see the trigger in action.
It will print "Hello, Temporal!" every 5 seconds.
How to run:
    make shell-app
    python  src/temporal/scripts/example_trigger.py

Or directly with Temporal CLI:
    uv run python src/temporal/scripts/example_trigger.py

Then open: http://localhost:8080/
"""

import uuid

import structlog
from temporalio.client import Client

from src.temporal.workflows.example_workflow import ExampleWorkflow

log = structlog.get_logger(__name__)


async def run_example_trigger(client: Client, name: str = "User") -> str:
    """Connects to the Temporal server, starts an instance of the ExampleWorkflow,
    and waits for the result."""

    workflow_id = f"example-{uuid.uuid4()}"

    # Run the workflow and wait for the result
    return await client.execute_workflow(
        ExampleWorkflow.run,
        name,
        id=workflow_id,
        task_queue="sandbox-task-queue",
    )
