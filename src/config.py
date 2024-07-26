"""Access and loading of deployment-specific settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.paths import DOTENV_PATH


class Settings(BaseSettings):
    """Application settings with defaults.

    See https://saurabh-kumar.com/python-dotenv/#file-format for .env format documentation.

    Attributes:
        openai_key: Key to access OpenAI API.
        openai_organization: Organization to access OpenAI API.
        openai_usable: Whether fields needed to access OpenAI API have been filled.
    """

    openai_key: str | None = None
    openai_organization: str | None = None

    @property
    def openai_usable(self) -> bool:  # noqa: D102
        return self.openai_key is not None and self.openai_organization is not None

    model_config = SettingsConfigDict(frozen=True, env_file=DOTENV_PATH)


settings = Settings()
