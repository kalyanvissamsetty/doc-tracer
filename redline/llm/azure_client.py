"""Azure-hosted OpenAI Chat Completions using the official SDK."""
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam

from redline.config import Settings


class ModelRequestError(RuntimeError):
    """A service error safe to display or trace."""


@dataclass(frozen=True)
class TokenRates:
    input_usd_per_million: float
    output_usd_per_million: float


@dataclass(frozen=True)
class ChatResult:
    completion: ChatCompletion
    input_tokens: int | None
    output_tokens: int | None
    estimated_usd: float | None


def image_part(
    content: bytes, mime_type: Literal["image/png", "image/jpeg"] = "image/png"
) -> ChatCompletionContentPartImageParam:
    encoded = b64encode(content).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def build_client(
    settings: Settings, token_provider: Callable[[], str] | None = None, *, max_retries: int = 2
) -> openai.OpenAI:
    endpoint = settings.azure_openai_endpoint
    version = settings.azure_openai_api_version
    options = {"timeout": 60.0, "max_retries": max_retries}
    if version == "v1":
        base_url = endpoint if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1"
        return openai.OpenAI(
            base_url=base_url + "/",
            api_key=token_provider or settings.azure_openai_api_key.get_secret_value(),
            **options,
        )
    auth = (
        {"azure_ad_token_provider": token_provider}
        if token_provider is not None
        else {"api_key": settings.azure_openai_api_key.get_secret_value()}
    )
    return openai.AzureOpenAI(azure_endpoint=endpoint, api_version=version, **auth, **options)


class AzureChat:
    def __init__(
        self, settings: Settings, *, client: openai.OpenAI | None = None,
        rates: dict[str, TokenRates] | None = None,
    ):
        self.client = client if client is not None else build_client(settings)
        self.deployments = {
            "agent": settings.azure_openai_deployment_agent,
            "vision": settings.azure_openai_deployment_vision,
        }
        self.rates = rates or {}

    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        role: Literal["agent", "vision"] = "agent",
        tools: list[ChatCompletionToolParam] | None = None,
        max_completion_tokens: int | None = None,
        response_format: ResponseFormat | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ChatResult:
        deployment = self.deployments[role]
        try:
            completion = self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                tools=tools if tools else openai.Omit(),
                response_format=response_format if response_format is not None else openai.Omit(),
                temperature=temperature if temperature is not None else openai.Omit(),
                parallel_tool_calls=parallel_tool_calls if parallel_tool_calls is not None else openai.Omit(),
                max_completion_tokens=(
                    max_completion_tokens if max_completion_tokens is not None else openai.Omit()
                ),
            )
        except openai.APIStatusError as exc:
            raise ModelRequestError(f"Azure OpenAI request failed (HTTP {exc.status_code}).") from None
        except openai.APIError:
            raise ModelRequestError("Azure OpenAI request failed. Check connectivity and service configuration.") from None
        usage = completion.usage
        rates = self.rates.get(deployment)
        estimate = None
        if usage is not None and rates is not None:
            estimate = (
                usage.prompt_tokens * rates.input_usd_per_million
                + usage.completion_tokens * rates.output_usd_per_million
            ) / 1_000_000
        return ChatResult(
            completion=completion,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            estimated_usd=estimate,
        )
