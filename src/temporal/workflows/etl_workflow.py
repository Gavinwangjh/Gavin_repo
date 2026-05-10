# with workflow.unsafe.imports_passed_through():
# from src.temporal.activities.extract_organisations import extract_organisations


# @workflow.defn
# class ETLWorkflow:
# @workflow.run
# async def run(self, source: str):
# extracted_data = await workflow.execute_activity(
#  extract_organisations,
#  source,
#   start_to_close_timeout=timedelta(seconds=120),
# )

# transformed_data = await workflow.execute_activity(
#  transform_organisations,
#  extracted_data,
#  start_to_close_timeout=timedelta(seconds=120),

# )
# return transformed_data
# loaded_data = await workflow.execute_activity(
#  load_organisations,
#  transformed_data,
# start_to_close_timeout=timedelta(seconds=120),
# )

# return loaded_data


# in order to make commit work this part needed to be commented ate this stage
