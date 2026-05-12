"""
Centralized module for creating and managing the Temporal client connection.
"""

from temporalio.client import Client

from config.settings import settings


async def get_temporal_client() -> Client:
    """Create and return a Temporal client."""

    return await Client.connect(settings.temporal_host_local)
