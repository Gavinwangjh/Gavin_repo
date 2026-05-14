from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities.extract_organisations import extract_organisations
    from src.temporal.activities.transform_organisations import transform_organisations
    from src.temporal.activities.enrich_websites import enrich_websites
    from src.temporal.activities.discover_websites import discover_websites
    from src.temporal.activities.enrich_sustainability import enrich_sustainability
    from src.temporal.activities.load_organisations import load_organisations


@workflow.defn
class ETLWorkflow:
    @workflow.run
    async def run(self, params: dict):
        source = params.get("source")
        limit = params.get("limit", 10)
        name_keyword = params.get("name_keyword")

        extracted_data = await workflow.execute_activity(
            extract_organisations,
            {
                "source": source,
                "limit": limit,
                "name_keyword": name_keyword,
            },
            start_to_close_timeout=timedelta(seconds=120),
        )

        website_data = await workflow.execute_activity(
            discover_websites,
            extracted_data,
            start_to_close_timeout=timedelta(seconds=120),
        )

        website_enriched_data = await workflow.execute_activity(
            enrich_websites,
            website_data,
            start_to_close_timeout=timedelta(seconds=120),
        )

        sustainability_data = await workflow.execute_activity(
            enrich_sustainability,
            website_enriched_data,
            start_to_close_timeout=timedelta(seconds=120),
        )

        transformed_data = await workflow.execute_activity(
            transform_organisations,
            sustainability_data,
            start_to_close_timeout=timedelta(seconds=120),
        )

        return transformed_data