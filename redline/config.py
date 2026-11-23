"""Load application settings without exposing credential values in errors."""
from collections.abc import Mapping
from pathlib import Path
import os
from urllib.parse import urlsplit
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator


class ConfigurationError(ValueError):
    pass


class JobDefaults(BaseModel):
    max_agent_steps: int = Field(default=60, gt=0)
    max_job_cost_usd: float = Field(default=5.0, gt=0, allow_inf_nan=False)


class CloudSettings(BaseModel):
    azure_storage_connection_string: SecretStr | None = None
    azure_storage_account_url: str | None = None
    azure_storage_container: str | None = None
    applicationinsights_connection_string: SecretStr | None = None
    redline_data_dir: Path = Path(".redline/web")

    @field_validator("azure_storage_account_url")
    @classmethod
    def validate_storage_url(cls, value: str | None) -> str | None:
        return _https_endpoint(value) if value else None


def _https_endpoint(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("must be an HTTPS endpoint without credentials, query or fragment")
    return value.rstrip("/")


class Settings(JobDefaults):
    azure_openai_endpoint: str
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "v1"
    azure_openai_deployment_agent: str
    azure_openai_deployment_vision: str
    azure_docintel_endpoint: str
    azure_docintel_key: SecretStr | None = None
    azure_auth_mode: Literal["key", "entra"] = "key"
    agent_context_tokens: int | None = Field(default=None, gt=0)
    agent_input_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    agent_output_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_context_tokens: int | None = Field(default=None, gt=0)
    vision_input_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    vision_output_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    docintel_layout_usd_per_page: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("azure_openai_endpoint", "azure_docintel_endpoint")
    @classmethod
    def https_endpoint(cls, value: str) -> str:
        return _https_endpoint(value)


def _load(model, dotenv_path, streamlit_secrets):

    values = dict(dotenv_values(dotenv_path))
    if streamlit_secrets is not None:
        for field in model.model_fields:
            name = field.upper()
            if name in streamlit_secrets:
                values[name] = streamlit_secrets[name]
    values.update(os.environ)
    selected = {
        field: values[field.upper()]
        for field in model.model_fields
        if values.get(field.upper()) is not None and str(values[field.upper()]).strip()
    }
    missing = []
    if model is Settings and selected.get("azure_auth_mode", "key") == "key":
        missing = [f"{name.upper()} (missing)" for name in ("azure_openai_api_key", "azure_docintel_key")
                   if name not in selected]
    try:
        result = model.model_validate(selected)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_input=False, include_url=False, include_context=False):
            name = str(error["loc"][0]).upper()
            problem = "missing" if error["type"] == "missing" else "invalid value"
            problems.append(f"{name} ({problem})")
        raise ConfigurationError("Check environment settings: " + ", ".join(problems + missing)) from None
    if missing:
        raise ConfigurationError("Check environment settings: " + ", ".join(missing))
    return result


def load_settings(
    dotenv_path: str | Path = ".env", *, streamlit_secrets: Mapping[str, object] | None = None
) -> Settings:
    return _load(Settings, dotenv_path, streamlit_secrets)


def load_job_defaults(
    dotenv_path: str | Path = ".env", *, streamlit_secrets: Mapping[str, object] | None = None
) -> JobDefaults:
    return _load(JobDefaults, dotenv_path, streamlit_secrets)


def load_cloud_settings(*, streamlit_secrets: Mapping[str, object] | None = None) -> CloudSettings:
    return _load(CloudSettings, ".env", streamlit_secrets)
