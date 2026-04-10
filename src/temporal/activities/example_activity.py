import structlog
from temporalio import activity

log = structlog.get_logger(__name__)


@activity.defn
async def example_activity(name: str) -> str:
    """Example activity that simulates processing a task.
    Exchange the logic with real processing (e.g., call an external API, perform a calculation, etc.)
    """
    log.info("Executing example_activity", name=name)
    # Simulate some processing (e.g., call an external API, perform a calculation, etc.)
    result = f"Hello: {name}. Workflow executed."
    log.info("example_activity completed", result=result)
    return result
