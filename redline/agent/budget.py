"""Preflight every Azure call using a deployment context limit and configured rates."""
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from redline.llm.azure_client import AzureChat, ChatResult, TokenRates
from redline.tracing import Tracer

MILLION = Decimal(1_000_000)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class DeploymentLimit:
    # Must cover the deployment's full input context, including images and schemas.
    input_tokens: int
    rates: TokenRates

    def __post_init__(self):
        if not isinstance(self.input_tokens, int) or self.input_tokens <= 0:
            raise ValueError("Deployment context token limit must be a positive integer.")
        for amount in (self.rates.input_usd_per_million, self.rates.output_usd_per_million):
            number = Decimal(str(amount))
            if not number.is_finite() or number < 0:
                raise ValueError("Token rates must be finite and nonnegative.")

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            input_tokens * Decimal(str(self.rates.input_usd_per_million))
            + output_tokens * Decimal(str(self.rates.output_usd_per_million))
        ) / MILLION


@dataclass
class Budget:
    max_steps: int
    max_usd: Decimal
    steps: int = 0
    spent: Decimal = Decimal(0)
    input_tokens: int = 0
    output_tokens: int = 0
    docintel_pages: int = 0

    def __post_init__(self):
        if self.max_steps <= 0 or not self.max_usd.is_finite() or self.max_usd <= 0:
            raise ValueError("Step and cost limits must be positive and finite.")

    def step(self) -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded("Comparison reached its step limit before delivery.")
        self.steps += 1

    def reserve(self, amount: Decimal) -> None:
        if self.spent + amount > self.max_usd:
            raise BudgetExceeded("The next Azure call would exceed the configured cost budget.")
        self.spent += amount


class MeteredChat:
    """The agent and nested vision tools share this single budget."""

    def __init__(self, chat: AzureChat, budget: Budget, limits: dict[str, DeploymentLimit], tracer: Tracer):
        if chat.client.max_retries != 0:
            raise ValueError("Budgeted Azure calls require SDK retries to be disabled.")
        self.chat_client = chat
        self.budget = budget
        self.limits = limits
        self.tracer = tracer

    def chat(
        self, messages, *, role: Literal["agent", "vision"] = "agent",
        max_completion_tokens: int, **kwargs,
    ) -> ChatResult:
        limit = self.limits[role]
        reserve = limit.cost(limit.input_tokens, max_completion_tokens)
        self.budget.step()
        self.budget.reserve(reserve)
        self.tracer.record(
            "model_call", role=role, step=self.budget.steps, reserved_usd=float(reserve), spent_usd=float(self.budget.spent),
            message_count=len(messages), max_input_tokens=limit.input_tokens,
            max_completion_tokens=max_completion_tokens,
        )
        try:
            result = self.chat_client.chat(
                messages, role=role, max_completion_tokens=max_completion_tokens, **kwargs,
            )
        except Exception:
            self.tracer.record("model_error", role=role, spent_usd=float(self.budget.spent))
            raise
        if result.input_tokens is not None and result.output_tokens is not None:
            self.budget.input_tokens += result.input_tokens
            self.budget.output_tokens += result.output_tokens
            self.budget.spent += limit.cost(result.input_tokens, result.output_tokens) - reserve
            if result.input_tokens > limit.input_tokens or result.output_tokens > max_completion_tokens:
                raise BudgetExceeded("Azure usage exceeded the configured deployment bound. Check the context limit.")
        self.tracer.record(
            "model_result", role=role, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            spent_usd=float(self.budget.spent),
            finish_reasons=[choice.finish_reason for choice in result.completion.choices],
            tool_call_count=sum(len(choice.message.tool_calls or ()) for choice in result.completion.choices),
        )
        return result
