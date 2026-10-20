from typing import Any

from pydantic import BaseModel, Field, field_validator

from onyx.tools.tool_implementations.web_search.models import WebContentProviderConfig
from shared_configs.enums import WebContentProviderType, WebSearchProviderType


def _stringify_scalar(value: object) -> object:
    """Pydantic v2 does not coerce bool/number to str, but provider configs may
    carry JSON booleans and numbers: the readers parse both spellings (see
    _parse_bool_config in web_search/providers.py)."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return value


def _stringify_config_values(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {key: _stringify_scalar(item) for key, item in value.items()}


class WebSearchProviderView(BaseModel):
    id: int
    name: str
    provider_type: WebSearchProviderType
    is_active: bool
    config: dict[str, str] | None
    masked_api_key: str | None = Field(
        default=None,
        description="Masked API key for display (e.g. 'sk-a...z456'). None means no key stored.",
    )

    @field_validator("config", mode="before")
    @classmethod
    def _coerce_config_values(cls, value: object) -> object:
        return _stringify_config_values(value)


class WebSearchProviderUpsertRequest(BaseModel):
    id: int | None = Field(default=None, description="Existing provider ID to update.")
    name: str
    provider_type: WebSearchProviderType
    config: dict[str, str] | None = None
    api_key: str | None = Field(
        default=None,
        description="API key for the provider. Only required when creating or updating credentials.",
    )
    api_key_changed: bool = Field(
        default=False,
        description="Set to true when providing a new API key for an existing provider.",
    )
    activate: bool = Field(
        default=False,
        description="If true, sets this provider as the active one after upsert.",
    )

    @field_validator("config", mode="before")
    @classmethod
    def _coerce_config_values(cls, value: object) -> object:
        return _stringify_config_values(value)


class WebContentProviderView(BaseModel):
    id: int
    name: str
    provider_type: WebContentProviderType
    is_active: bool
    config: WebContentProviderConfig | None
    masked_api_key: str | None = None


class WebContentProviderUpsertRequest(BaseModel):
    id: int | None = None
    name: str
    provider_type: WebContentProviderType
    config: WebContentProviderConfig | None = None
    api_key: str | None = None
    api_key_changed: bool = False
    activate: bool = False


class WebSearchProviderTestRequest(BaseModel):
    provider_type: WebSearchProviderType
    api_key: str | None = Field(
        default=None,
        description="API key for testing. If not provided, use_stored_key must be true.",
    )
    use_stored_key: bool = Field(
        default=False,
        description="If true, use the stored API key for this provider type instead of api_key.",
    )
    config: dict[str, Any] | None = None


class WebContentProviderTestRequest(BaseModel):
    provider_type: WebContentProviderType
    api_key: str | None = Field(
        default=None,
        description="API key for testing. If not provided, use_stored_key must be true.",
    )
    use_stored_key: bool = Field(
        default=False,
        description="If true, use the stored API key for this provider type instead of api_key.",
    )
    config: WebContentProviderConfig
