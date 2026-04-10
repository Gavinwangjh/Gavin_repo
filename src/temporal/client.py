"""
Centralized module for creating and managing the Temporal client connection.
"""

from temporalio.client import Client

from config.settings import settings


async def get_temporal_client() -> Client:
    """Create and return a Temporal client connected to the Temporal server specified in settings."""
    # settings.TEMPORAL_HOST should be eg. "localhost:7233" ili "temporal:7233"
    return await Client.connect(settings.temporal_host)
