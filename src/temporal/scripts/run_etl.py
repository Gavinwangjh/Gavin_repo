import argparse
import asyncio
import uuid

from temporalio.client import Client

from src.temporal.workflows.etl_workflow import ETLWorkflow


async def run(source: str, limit: int, name_keyword: str | None = None):
    client = await Client.connect("temporal:7233", namespace="default")

    workflow_params = {
        "source": source,
        "limit": limit,
        "name_keyword": name_keyword,
    }

    result = await client.execute_workflow(
        ETLWorkflow.run,
        workflow_params,
        id=f"etl-{source}-{uuid.uuid4()}",
        task_queue="sandbox-task-queue",
    )

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL workflow")

    parser.add_argument(
        "--source",
        choices=["au", "uk", "us_sec", "au_wgea"],
        required=True,
        help="Data source to run ETL for: au, uk, us_sec",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of records to extract",
    )

    parser.add_argument(
        "--name-keyword",
        default=None,
        help="Optional organisation name keyword filter",
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            source=args.source,
            limit=args.limit,
            name_keyword=args.name_keyword,
        )
    )
