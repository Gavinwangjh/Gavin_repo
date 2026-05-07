import asyncio
import uuid

import structlog
from temporalio.client import Client

from src.temporal.workflows.etl_workflow import ETLWorkflow

log = structlog.get_logger(__name__)


async def run_etl_trigger(client: Client, source: str = "au") -> dict:
    """Starts the ETL workflow and waits for the result."""

    workflow_id = f"etl-{source}-{uuid.uuid4()}"

    print(f"Starting ETL workflow: {workflow_id}")

    result = await client.execute_workflow(
        ETLWorkflow.run,
        source,
        id=workflow_id,
        task_queue="sandbox-task-queue",
    )

    print(f"ETL workflow completed: {workflow_id}")
    return result


async def main() -> None:
    client = await Client.connect("temporal:7233", namespace="default")
    result = await run_etl_trigger(client, source="au")
    print(f"Workflow result: {result}")


if __name__ == "__main__":
    asyncio.run(main())