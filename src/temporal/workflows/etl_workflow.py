from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities.extract_organisations import extract_organisations


@workflow.defn
class ETLWorkflow:
    @workflow.run
    async def run(self, source: str):

        return await workflow.execute_activity(
            extract_organisations,
            source,
            start_to_close_timeout=timedelta(seconds=120),
        )
