"""
Temporal worker for ETL pipeline.
"""

import asyncio

import structlog
from temporalio.worker import Worker

from src.temporal.activities.discover_websites import (
    discover_websites,
)
from src.temporal.activities.enrich_sustainability import (
    enrich_sustainability,
)
from src.temporal.activities.enrich_websites import (
    enrich_websites,
)

# =========================================================
# Activities
# =========================================================
from src.temporal.activities.extract_organisations import (
    extract_organisations,
)
from src.temporal.activities.transform_organisations import (
    transform_organisations,
)
from src.temporal.client import get_temporal_client

# =========================================================
# Workflows
# =========================================================
from src.temporal.workflows.etl_workflow import ETLWorkflow

log = structlog.get_logger(__name__)


async def run_worker() -> None:

    # =====================================================
    # Temporal client
    # =====================================================

    client = await get_temporal_client()

    # =====================================================
    # Worker
    # =====================================================

    worker = Worker(
        client,
        # 👇 Temporal task queue
        task_queue="sandbox-task-queue",
        # 👇 register workflows
        workflows=[
            ETLWorkflow,
        ],
        # 👇 register activities
        activities=[
            extract_organisations,
            enrich_websites,
            discover_websites,
            enrich_sustainability,
            transform_organisations,
        ],
    )

    log.info(
        "ETL Worker started",
        task_queue="sandbox-task-queue",
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
