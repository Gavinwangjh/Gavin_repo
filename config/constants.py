"""
Project-wide constants.

- Define string literals, numeric constants, and default values here.
"""

from __future__ import annotations

import pathlib
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel

_CURRENT_DIR = pathlib.Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent
GENERIC_TYPE = TypeVar("GENERIC_TYPE", bound=BaseModel)


class AppEnv(StrEnum):
    """
    Defines the execution environments for the application.

    Used to toggle environment-specific logic such as logging levels,
    storage providers (local vs. cloud), and database connection strings.
    """

    LOCAL = "local"
    DEVELOPMENT = "development"
    UAT = "uat"
    PRODUCTION = "production"
