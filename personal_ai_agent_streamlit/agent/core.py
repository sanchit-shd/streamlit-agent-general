
"""OpenRouter-powered personal AI agent."""

import os
import logging
from typing import Iterable

import streamlit as st
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class AgentError(Exception):
    pass


class AgentConfigurationError(AgentError):
    pass


SYSTEM_INSTRUCTIONS = """
You are a helpful personal AI assistant.

Your responsibilities:
1. Answer questions accurately.
2. Explain technical concepts clearly.
3. Provide practical examples when useful.
4. Structure complex answers using headings and bullet points.
5. If you are unsure, clearly mention it.
"""


def get_config(name: str, default=None):
    """Read configuration from Streamlit Secrets or environment."""

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    return value or os.getenv(name, default)


def build_client():

    api_key = get_config("OPENAI_API_KEY")

    if not api_key:
        raise AgentConfigurationError(
            "OpenRouter API key is missing. "
            "Configure OPENAI_API_KEY in Streamlit Secrets."
        )

    base_url = get_config(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    timeout = float(get_config("OPENAI_TIMEOUT_SECONDS", "45"))
    retries = int(get_config("OPENAI_MAX_RETRIES", "2"))

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=retries,
    )


def normalize_history(history: Iterable = None):

    messages = []

    if not history:
        return messages

    for item in history:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({
                "role": role,
                "content": content.strip()
            })

    return messages[-20:]


def run_agent(question, conversation_history=None):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    model = get_config(
        "OPENAI_MODEL",
        "openai/gpt-oss-120b"
    )

    client = build_client()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTIONS
        }
    ]

    messages.extend(
        normalize_history(conversation_history)
    )

    messages.append({
        "role": "user",
        "content": question.strip()
    })

    try:

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise AgentError("Model returned an empty response.")

        return answer.strip()

    except AuthenticationError as exc:
        logger.error("OpenRouter authentication failed.")
        raise AgentError(
            "OpenRouter authentication failed. "
            "Verify your API key and provider configuration."
        ) from exc

    except RateLimitError as exc:
        logger.error("OpenRouter rate limit or quota exceeded.")
        raise AgentError(
            "OpenRouter rate limit or credit limit reached. "
            "Check your OpenRouter account."
        ) from exc

    except APITimeoutError as exc:
        logger.error("OpenRouter request timed out.")
        raise AgentError(
            "The model request timed out. Please try again."
        ) from exc

    except APIConnectionError as exc:
        logger.error("Could not connect to OpenRouter.")
        raise AgentError(
            "Could not connect to OpenRouter. Check your connection."
        ) from exc

    except APIStatusError as exc:
        logger.error(
            "OpenRouter API returned HTTP %s.",
            exc.status_code
        )
        raise AgentError(
            f"OpenRouter returned an API error (HTTP {exc.status_code})."
        ) from exc
