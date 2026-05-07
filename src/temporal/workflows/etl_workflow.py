from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities.extract_organisations import extract_organisations
    from src.temporal.activities.transform_organisations import transform_organisations


@workflow.defn
class ETLWorkflow:
    @workflow.run
    async def run(self, source: str):
        extracted_data = await workflow.execute_activity(
            extract_organisations,
            source,
            start_to_close_timeout=timedelta(seconds=120),
        )

        transformed_data = await workflow.execute_activity(
            transform_organisations,
            extracted_data,
            start_to_close_timeout=timedelta(seconds=120),
        )

        return transformed_data
