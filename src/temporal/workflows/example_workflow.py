"""
Example Temporal workflow demonstrating basic workflow and activity usage.

- Define workflows as async functions.
- Activities are tasks that can be called from workflows.
- Users can copy and modify this file to create their own workflows.
"""

from datetime import timedelta

import structlog
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities.example_activity import example_activity

log = structlog.get_logger(__name__)


@workflow.defn
class ExampleWorkflow:
    """
    Example workflow that calls an activity to generate a greeting message.
    Workflow calls an activity and waits for the result before returning it.

    Structure of every workflow:
    1.@workflow.defn on the class definition
    2. @workflow.run on the method that will be executed when the workflow is started
    3. Call activities using "await workflow.execute_activity" and specify timeouts
    4. Always define start_to_close_timeout for activities to prevent them from running
    indefinitely

    """

    @workflow.run
    async def run(self, name: str):
        # Call the activity and wait for the result
        # Note: Temporal always uses "execute_activity" to call activities,
        # even if they are defined in the same file
        log.info("example_workflow_started", name=name)
        result = await workflow.execute_activity(
            example_activity, name, start_to_close_timeout=timedelta(seconds=10)
        )
        log.info("example_workflow_completed", result=result)
        return result
