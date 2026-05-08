#!/usr/bin/env python3
"""
Run the example workflow and print the result.
This script connects to Temporal, triggers the workflow, and waits for the result.
"""

import asyncio
import sys
import uuid

from src.temporal.client import get_temporal_client
from src.temporal.workflows.example_workflow import ExampleWorkflow


async def main():
    """Main entry point to run the example workflow."""
    # Connect to Temporal server
    client = await get_temporal_client()

    print("✅ Connected to Temporal server")
    print("🚀 Starting ExampleWorkflow...")

    try:
        # Execute the workflow with a unique ID
        workflow_id = f"example-workflow-{uuid.uuid4()}"
        result = await client.execute_workflow(
            ExampleWorkflow.run,
            "User",
            id=workflow_id,
            task_queue="sandbox-task-queue",
        )

        print("✨ Workflow completed successfully!")
        print(f"📝 Result: {result}")

    except Exception as e:
        print(f"❌ Error executing workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
