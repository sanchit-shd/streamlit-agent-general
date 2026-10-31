# Personal AI Agent — Streamlit starter

A small, deployable Streamlit chat application with agent logic separated from the UI.

## 1. Run locally

Requires Python 3.10+.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (do not commit this file):

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_TIMEOUT_SECONDS = "45"
OPENAI_MAX_RETRIES = "2"
```

Start the app:

```bash
streamlit run app.py
```

## 2. Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. Open https://share.streamlit.io/ and create a new app.
3. Select your repository, branch, and `app.py` as the main file.
4. In the app's settings, open **Secrets** and add:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_TIMEOUT_SECONDS = "45"
OPENAI_MAX_RETRIES = "2"
```

5. Save and deploy.

Never commit your API key or the real `secrets.toml` file to GitHub.

## 3. Connect your own agent

The UI imports one function:

```python
from agent.core import run_agent
answer = run_agent(question, conversation_history)
```

Replace the implementation in `agent/core.py` with your own agent framework or orchestration. Keep the same function contract and return a string. For example, your implementation can invoke LangGraph, an OpenAI Agent, custom Python tools, or MCP clients.

## 4. Important production notes

- This starter uses session state for chat history. It is not durable storage and does not share history across devices or users.
- For multi-user production use, add authentication, per-user authorization, persistent database storage, request quotas, and audit logging.
- Do not put API keys in source code, browser code, or Git.
- Add only trusted tools/MCP servers. Validate tool inputs and require confirmation for consequential actions.
- The sample is an LLM chat foundation; it does not yet include autonomous tool execution or long-term memory.
