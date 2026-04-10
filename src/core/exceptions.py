"""
Custom exceptions for the application.
"""

from __future__ import annotations


class AppError(Exception):
    """Base exception for application errors."""

    def __init__(self, *, message: str, stage: str = "GENERAL") -> None:
        self.message = message
        self.stage = stage
        super().__init__(self.message)


class DatabaseError(AppError):
    """Base exception for database errors."""

    def __init__(self, *, message: str) -> None:
        super().__init__(message=message, stage="DATABASE")


class DatabaseConnectionError(DatabaseError):
    """Raised when a connection to the database cannot be established."""

    pass
