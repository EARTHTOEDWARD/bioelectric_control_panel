from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration for BCP.

    Env prefix: BCP_
    Load from .env by default.
    """

    model_config = SettingsConfigDict(env_prefix="BCP_", env_file=".env", extra="ignore")

    maes_project_path: Path = Path("./data/maes")
    control_panel_data_import: Path = Path("./data/incoming")
    control_panel_data_export: Path = Path("./data/outgoing")

    watch_interval: int = 5
    shared_format_version: str = "1.0"

    def ensure_dirs(self) -> None:
        self.control_panel_data_import.mkdir(parents=True, exist_ok=True)
        self.control_panel_data_export.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

