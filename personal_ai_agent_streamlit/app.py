import logging
import streamlit as st

from agent.core import AgentConfigurationError, AgentError, run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Personal AI Assistant")
st.caption("A private chat UI powered by your Python agent.")

# Session state is per browser session. Use a database for durable history.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Chat controls")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Conversation history is held in this browser session only.")
    st.divider()
    st.markdown("**Configuration**")
    st.caption("Set `OPENAI_API_KEY` in Streamlit Cloud → App → Settings → Secrets.")

# Render existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything…")

if prompt:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a question.")
        st.stop()

    # Keep a bounded context window. The agent receives previous turns,
    # excluding the current prompt which is passed separately.
    prior_messages = st.session_state.messages[-20:]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = run_agent(
                    question=prompt,
                    conversation_history=prior_messages,
                )
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except AgentConfigurationError as exc:
                logger.warning("Agent configuration problem: %s", exc)
                st.error(str(exc))
            except AgentError:
                logger.exception("Agent request failed")
                st.error(
                    "I couldn't complete that request. Please try again. "
                    "If the problem continues, check the application logs."
                )
            except Exception:
                # Avoid displaying stack traces or secrets to end users.
                logger.exception("Unexpected application error")
                st.error("An unexpected error occurred. Please try again.")
