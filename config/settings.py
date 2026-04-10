"""
Application settings.

- Central place to store configuration values.
- Can be extended to read from environment variables or YAML files.
"""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import _ROOT_DIR, AppEnv


class Settings(BaseSettings):
    PROJECT_NAME: str
    APP_ENV: AppEnv = Field(default=AppEnv.LOCAL, validation_alias="AZURE_ENV")

    # PostgreSQL database settings
    DB_USER: str = Field(..., description="Database username.")
    DB_PASSWORD: str = Field(..., description="Database password.")
    DB_HOST: str = Field(..., description="Database host.")
    DB_NAME: str = Field(..., description="Database host URL.")
    DB_PORT: int = Field(..., description="Database port.")

    # Temporal server settings
    TEMPORAL_PORT: int = Field(..., description="Temporal server port.")
    TEMPORAL_NAMESPACE: str = Field(..., description="Temporal namespace to use.")
    POSTGRES_SEEDS: str = Field(
        ..., description="Comma-separated list of PostgreSQL seed hosts for Temporal server."
    )

    @computed_field
    @property
    def db_url(self) -> str:
        """
        Constructs a PostgreSQL connection URL.

        Format:
            postgresql://<user>:<password>@<host>:<port>/<database>

        Components:
        - user (DB_USER): Username used to authenticate with the database.
        - password (DB_PASSWORD): Password for the database user.
        - host (DB_HOST): Address of the database server.
            - Use "postgres" when running inside Docker (service name).
            - Use "localhost" when connecting to a local database.
        - port (DB_PORT): Port on which the database is exposed (default: 5432).
        - database (DB_NAME): Name of the target database.

        Example:
            postgresql://sandbox:secret@postgres:5432/mydb
        """
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@postgres:{self.DB_PORT}/{self.DB_NAME}"

    @computed_field
    @property
    def async_db_url(self) -> str:
        """Construct the asynchronous database URL."""
        return (
            f"postgresql+asyncpg://"
            f"{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @computed_field
    @property
    def temporal_host(self) -> str:
        """Temporal address inside docker network."""
        return f"temporal:{self.TEMPORAL_PORT}"

    @computed_field
    @property
    def temporal_host_local(self) -> str:
        """Temporal address on local PC."""
        return f"localhohst:{self.TEMPORAL_PORT}"

    model_config = SettingsConfigDict(
        env_file=_ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
