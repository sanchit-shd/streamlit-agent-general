
"""Google Gemini-powered personal AI agent (native google-genai SDK)."""

import os
import logging
from typing import Iterable

import streamlit as st
from google import genai
from google.genai import errors, types

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
    """Build a Google GenAI client using the configured API key."""

    api_key = get_config("GEMINI_API_KEY")

    if not api_key:
        raise AgentConfigurationError(
            "Google Gemini API key is missing. "
            "Configure GEMINI_API_KEY in Streamlit Secrets."
        )

    # Timeout (in seconds) applied to each HTTP request to Google's API.
    timeout = float(get_config("GEMINI_TIMEOUT_SECONDS", "45"))

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout),
    )


def normalize_history(history: Iterable = None):
    """Convert chat history dicts into Gemini Content objects."""

    contents = []

    if not history:
        return contents

    for item in history:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = item.get("content")

        # Gemini uses "model" for the AI's turns, not "assistant",
        # so translate the role before sending it to the API.
        if role == "assistant":
            role = "model"

        if (
            role in ("user", "model")
            and isinstance(text, str)
            and text.strip()
        ):
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=text.strip())],
                )
            )

    # Keep only the most recent turns to stay within context limits.
    return contents[-20:]


def run_agent(question, conversation_history=None):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Google Gemini models:
    #   - gemini-2.5-flash-lite  (fastest, cheapest)
    #   - gemini-2.5-pro        (most capable)
    # Override via GEMINI_MODEL in Streamlit Secrets or environment.
    model = get_config(
        "GEMINI_MODEL",
        "gemini-2.5-flash-lite"
    )

    client = build_client()

    # Full conversation: prior turns plus the new question.
    contents = normalize_history(conversation_history)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=question.strip())],
        )
    )

    try:

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
            ),
        )

        # response.text raises ValueError when the response contains
        # no text parts (e.g. the reply was safety-blocked or empty).
        try:
            answer = response.text
        except ValueError:
            answer = None

        if not answer:
            raise AgentError("Model returned an empty response.")

        return answer.strip()

    except errors.ClientError as exc:
        # 4xx errors: invalid key, unknown model, quota exhausted, etc.
        logger.error(
            "Gemini client error (HTTP %s): %s", exc.code, exc.message
        )
        raise AgentError(
            f"Gemini returned a client error (HTTP {exc.code}). "
            "Verify your API key, model name, and quota."
        ) from exc

    except errors.ServerError as exc:
        # 5xx errors: transient issues on Google's side.
        logger.error(
            "Gemini server error (HTTP %s): %s", exc.code, exc.message
        )
        raise AgentError(
            f"Gemini server error (HTTP {exc.code}). Please try again."
        ) from exc

    except errors.APIError as exc:
        # Catch-all for any other Gemini API failure (incl. timeouts).
        logger.error("Gemini API error: %s", exc)
        raise AgentError(
            "The Gemini request failed. Please try again."
        ) from exc
