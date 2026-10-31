"""Core agent entry point.

Keep orchestration, prompts, tools, and model calls out of the Streamlit UI.
Replace or extend run_agent() with your own agent framework when ready.
"""
from __future__ import annotations

import os
from typing import Iterable

import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


class AgentError(Exception):
    """Raised when the agent cannot complete a request."""


class AgentConfigurationError(AgentError):
    """Raised when required application configuration is missing."""


SYSTEM_INSTRUCTIONS = """You are a helpful personal AI assistant.
Answer clearly and accurately. If you are unsure, say so.
Use concise explanations and practical examples when useful."""


def _get_secret(name: str, default: str | None = None) -> str | None:
    """Read configuration from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


def _build_client() -> OpenAI:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise AgentConfigurationError(
            "OPENAI_API_KEY is not configured. Add it to Streamlit Cloud "
            "App settings → Secrets, or to your local .streamlit/secrets.toml."
        )

    try:
        timeout_seconds = float(_get_secret("OPENAI_TIMEOUT_SECONDS", "45"))
        max_retries = int(_get_secret("OPENAI_MAX_RETRIES", "2"))
    except (TypeError, ValueError) as exc:
        raise AgentConfigurationError(
            "OPENAI_TIMEOUT_SECONDS must be numeric and "
            "OPENAI_MAX_RETRIES must be an integer."
        ) from exc

    return OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _normalise_history(
    history: Iterable[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Accept only user/assistant messages with non-empty string content."""
    normalised: list[dict[str, str]] = []
    if not history:
        return normalised

    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            normalised.append({"role": role, "content": content.strip()})
    return normalised[-20:]


def run_agent(
    question: str,
    conversation_history: Iterable[dict[str, str]] | None = None,
) -> str:
    """Run the assistant and return a plain-text answer.

    `conversation_history` should contain only prior user/assistant turns.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    model = _get_secret("OPENAI_MODEL", "gpt-4.1-mini")
    client = _build_client()

    input_messages = _normalise_history(conversation_history)
    input_messages.append({"role": "user", "content": question.strip()})

    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=input_messages,
        )
        answer = (response.output_text or "").strip()
        if not answer:
            raise AgentError("The model returned an empty response.")
        return answer

    except (APIConnectionError, APITimeoutError) as exc:
        raise AgentError("Could not connect to the model provider.") from exc
    except APIStatusError as exc:
        # Keep provider details in logs, not in the UI.
        raise AgentError("The model provider rejected the request.") from exc
