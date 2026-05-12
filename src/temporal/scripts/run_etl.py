import asyncio
import uuid

from temporalio.client import Client

from src.temporal.workflows.etl_workflow import (
    ETLWorkflow,
)


async def run(source: str):
    client = await Client.connect("temporal:7233")
    
    # Dynamic workflow parameters
    

    workflow_params = {
        # data source
        "source": "au",
        #  control extraction size
        "limit": 5,
        # optional keyword filtering
        # "name_keyword": "care",
    }

    # =====================================================
    # Execute workflow
    # =====================================================

    result = await client.execute_workflow(
        ETLWorkflow.run,
        workflow_params,
        id=(f"etl-{workflow_params['source']}-{uuid.uuid4()}"),
        task_queue="sandbox-task-queue",
    )

    # =====================================================
    # Output result
    # =====================================================

    print(result)


# =========================================================
# Run ETL
# =========================================================

asyncio.run(run())
