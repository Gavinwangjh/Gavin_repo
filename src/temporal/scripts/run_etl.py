import asyncio
import uuid

from temporalio.client import Client

from src.temporal.workflows.etl_workflow import ETLWorkflow


async def run(source: str):
    # ✅ 先连接 Temporal
    client = await Client.connect("localhost:7233")

    result = await client.execute_workflow(
        ETLWorkflow.run,
        source,
        id=f"etl-{source}-{uuid.uuid4()}",
        task_queue="sandbox-task-queue",
    )

    print(result)


# ✅ 指定数据源
asyncio.run(run("au"))  # 改成 "au" / "uk"
