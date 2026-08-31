from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    app_name: str = "RAG API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"
    database_url: str | None = None
    database_host: str | None = None
    database_port: int = 5432
    database_user: str | None = None
    database_pass: SecretStr | None = None
    database_name: str | None = None
    db_ssl: bool = True
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_response_model: str = "gpt-5.4-mini"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "https://rag-front-jizp.onrender.com",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def use_psycopg_driver(cls, value: object) -> object:
        """Adapt standard PostgreSQL URLs to SQLAlchemy's Psycopg 3 dialect."""

        if not isinstance(value, str):
            return value

        if value.startswith("postgres://"):
            value = value.replace("postgres://", "postgresql://", 1)

        if value.startswith("postgresql://"):
            return value.replace(
                "postgresql://",
                "postgresql+psycopg://",
                1,
            )

        return value

    @property
    def sqlalchemy_database_url(self) -> URL:
        """Build a safe SQLAlchemy URL from either supported env format."""

        host = self.database_host
        user = self.database_user
        database = self.database_name

        password = (
            self.database_pass.get_secret_value()
            if self.database_pass is not None
            else None
        )

        if host and user and password and database:
            return URL.create(
                drivername="postgresql+psycopg",
                username=user,
                password=password,
                host=host,
                port=self.database_port,
                database=database,
                query={"sslmode": "require"} if self.db_ssl else {},
            )

        if self.database_url:
            return make_url(self.database_url)

        connection_parts = {
            "DATABASE_HOST": host,
            "DATABASE_USER": user,
            "DATABASE_PASS": password,
            "DATABASE_NAME": database,
        }

        missing = [
            name
            for name, value in connection_parts.items()
            if not value
        ]

        raise ValueError(
            "Database configuration is incomplete. Missing: "
            + ", ".join(missing)
        )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the application process."""

    return Settings()
